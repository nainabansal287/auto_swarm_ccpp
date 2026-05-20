#changes:
#1. If 1st drone flies at 6m, next at 7m, 8m so on. 
#2. Drones will perform mode RTL in sequence instead of LAND at final waypoint.

import argparse
import os
import sys
import time
import math
import threading
import main_height_diff #change if name of main_new file changes
import glob
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MAIN_SCRIPT  = 'main_height_diff.py'
MISSIONS_DIR = 'missions'
#DRONE_DELAY  = 60.0    # seconds between consecutive takeoffs - NOW REMOVED
TAKEOFF_ALT  = 6.0     # metres (relative / AGL)
MIN_HORIZ_DIST   = 2.0   # metres — trigger hold if closer than this
#MIN_ALTI_DIST = 2.0
AVOIDANCE_HOLD_S = 2.0   # seconds between re-checks while holding
LIFT             = 2.0  # alti increase in case of collision

CONNECT_TIMEOUT  = 30   # seconds — wait for first heartbeat
#ARM_TIMEOUT      = 30    # seconds
POLL_HZ = 4   

#Drone connection strings
DRONES: List[Tuple[int, str, int]] = [
    # (1-based index,  conn_str, sysid)
    (1,  'udpin:127.0.0.1:14550',  1),
    (2,  'udpin:127.0.0.1:14560',  2),
    ]



# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('swarm_mission.log', mode='w'),
    ],
)
log = logging.getLogger('swarm')

# ═══════════════════════════════════════════════════════════════════════════
#  DRONE STATE CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DroneState:
    index:           int
    sysid:           int
    conn_str:        str
    mission_wps:     list                         = field(default_factory=list)
    lawnmower_start: Optional[Tuple[float,float]] = None   # (lat, lon)

    # Live telemetry — written by owner thread, read by sibling threads
    lat:        float = 0.0
    lon:        float = 0.0
    launch_lat: float = 0.0
    launch_lon: float = 0.0
    alt_rel:    float = 0.0
    in_transit: bool  = True   # False after drone reaches lawnmower start WP
    done:       bool  = False
    error:      Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock) #ensures only 1 thread can access one function at a time

    def set_pos(self, lat: float, lon: float, alt: float): #setter function to alter lat,lon,alt
        with self._lock:
            self.lat, self.lon, self.alt_rel = lat, lon, alt

    def get_pos(self) -> Tuple[float, float, float]: #getter function to access lat,lon,alt
        with self._lock:
            return self.lat, self.lon, self.alt_rel

# ═══════════════════════════════════════════════════════════════════════════
#Haversine
# ═══════════════════════════════════════════════════════════════════════════
R_EARTH = 6_371_000.0

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ═══════════════════════════════════════════════════════════════════════════
#  WAYPOINT FILE PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_waypoint_file(filepath: str) -> List[Tuple[float, float, float]]:
    """Parse QGC WPL 110 file → list of (lat, lon, alt_rel), home excluded."""
    wps = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines[1:]:          # skip header row
            parts = line.strip().split('\t')
            if len(parts) < 11: #standard waypoint file has 12 columns of data, so if less than 11, means file is malformed
                continue
            seq = int(parts[0])
            cmd = int(parts[3])
            lat = float(parts[8])
            lon = float(parts[9])
            alt = float(parts[10])
            if seq == 0:
                continue               # skip home WP
            if cmd == 16:              # MAV_CMD_NAV_WAYPOINT
                wps.append((lat, lon, alt))
    except Exception as exc:
        log.error(f"parse_waypoint_file({filepath}): {exc}")
    return wps

# ═══════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL MAVLink HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def wait_heartbeat(mav, timeout: float = CONNECT_TIMEOUT):
    """Block until the first HEARTBEAT arrives from the autopilot."""
    log.info("    Waiting for HEARTBEAT …")
    msg = mav.wait_heartbeat(timeout=timeout)
    if msg is None:
        raise TimeoutError("No HEARTBEAT received within timeout")
    log.info(f"    HEARTBEAT received  sysid={mav.target_system}  "
             f"compid={mav.target_component}")


def request_streams(mav, hz: int = POLL_HZ):
    """Ask autopilot to stream position + status messages."""
    for stream_id in (
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
    ):
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            stream_id, hz, 1,
        )


def get_gps(mav, timeout: float = 2.0) -> Optional[Tuple[float, float, float]]:
    """Return (lat_deg, lon_deg, alt_rel_m) or None."""
    msg = mav.recv_match(type='GLOBAL_POSITION_INT',
                         blocking=True, timeout=timeout)
    if msg is None:
        return None
    return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0

def set_mode(mav,mode_id: int, timeout: float = 5.0):
    modes = { #note: this only applies to ardupilot, will have to change w px4
        0: 'STABILIZE',
        4: 'GUIDED',
        5: 'LOITER',
        6: 'RTL' ,
        9: 'LAND'
    }
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        heartbeat = mav.recv_match(type='HEARTBEAT', blocking = True, timeout = 1)
        if heartbeat and heartbeat.custom_mode == mode_id:
            return
    log.warning(f"    set_mode({modes[mode_id]}) — no confirmation within {timeout}s")

def send_command_long(mav, command: int,
                      p1=0.0, p2=0.0, p3=0.0,
                      p4=0.0, p5=0.0, p6=0.0, p7=0.0,
                      confirmation: int = 0):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        command, confirmation,
        p1, p2, p3, p4, p5, p6, p7,
    )


def is_armed(mav, timeout: float = 2.0) -> bool:
    hb = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
    if hb is None:
        return False
    return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

# ═══════════════════════════════════════════════════════════════════════════
#  MISSION UPLOAD  (MISSION_ITEM_INT handshake)
# ═══════════════════════════════════════════════════════════════════════════

def upload_mission(mav, wps: List[Tuple[float, float, float]], tag: str):
    total = len(wps) + 1   # +1 for the home item at seq=0

    # Send MISSION_COUNT to start the handshake
    mav.mav.mission_count_send(
        mav.target_system, mav.target_component,
        total,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    sent = 0
    t0   = time.time()
    while sent < total:
        req = mav.recv_match(type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'],
                             blocking=True, timeout=3)
        if req is None:
            continue

        seq = req.seq

        if seq == 0:
            # Home item — use vehicle's current position
            pos = get_gps(mav, timeout=3) #removed 'or (0.0, 0.0, 0.0)'
            lat_i = int(pos[0] * 1e7)
            lon_i = int(pos[1] * 1e7)
            alt_f = 0.0
            autocont = 0
        else:
            wp_lat, wp_lon, wp_alt = wps[seq - 1]
            lat_i = int(wp_lat * 1e7)
            lon_i = int(wp_lon * 1e7)
            alt_f = float(wp_alt)
            autocont = 1

        mav.mav.mission_item_int_send(
            mav.target_system, mav.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0,          # current
            autocont,
            0, 0, 0, 0, # param1-4
            lat_i, lon_i, alt_f,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        sent = seq + 1

    # Wait for MISSION_ACK
    ack = mav.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    if ack is None or ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"{tag} MISSION_ACK not accepted: {ack}")
    log.info(f"{tag} Mission uploaded  ({len(wps)} WPs)")

 # COLLISION AVOIDANCE
 # Collision avoidance helpers - 1. hold position 2.
 # 

def hold_position(mav, lat: float, lon: float, alt: float):
    """Command vehicle to hold its current position (GUIDED + position target)."""
    mav.mav.set_position_target_global_int_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,   # type_mask: only lat/lon/alt used
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0,              # vx, vy, vz (ignored)
        0, 0, 0,              # ax, ay, az (ignored)
        0, 0,                 # yaw, yaw_rate (ignored)
    )   

def find_lift_drone(states: List[DroneState],idx_a: int,idx_b: int) -> Tuple[int, int]:
    alt_a = states[idx_a].get_pos()[2]
    alt_b = states[idx_b].get_pos()[2]
    if alt_a >= alt_b:
        return idx_a, idx_b
    return idx_b, idx_a

def go_to(mav, lat: float, lon: float, alt: float):
    mav.mav.set_position_target_global_int_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000111111111000,          # type_mask: lat/lon/alt only
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0,
        0, 0, 0,
        0, 0,
    )

def collision_check(states: List[DroneState], curr_idx: int) -> Optional[Tuple[int, int, int]]:
    """
    Checks for horizontal collision risk between current drone and all others.
    Returns (curr_idx, other_idx, lift_idx) if risk exists, else None.
    lift_idx is the index of whichever drone is currently at higher altitude.
    """
    current = states[curr_idx]
    curr_lat, curr_lon, curr_alt = current.get_pos()

    for i, other in enumerate(states):
        if i == curr_idx or not other.in_transit:
            continue
        o_lat, o_lon, o_alt = other.get_pos()
        d = haversine(curr_lat, curr_lon, o_lat, o_lon)
        if d < MIN_HORIZ_DIST:
            log.info(f"D{current.index} & D{other.index}  "
                     f"horiz={d:.2f}m  alti_diff={abs(curr_alt - o_alt):.2f}m")
            lift_idx = curr_idx if curr_alt >= o_alt else i
            return curr_idx, i, lift_idx
    return None

def alti_check(states: List[DroneState], curr_idx: int) -> Optional[Tuple[int, int, int]]:
    current = states[curr_idx]
    curr_lat, curr_lon, curr_alt = current.get_pos()

    for i, other in enumerate(states):
        if i == curr_idx or not other.in_transit:
            continue
        o_lat, o_lon, o_alt = other.get_pos()
        d = haversine(curr_lat, curr_lon, o_lat, o_lon)
        if d < MIN_HORIZ_DIST:
            return abs(curr_alt - o_alt)
    

def read_launch_position(mav, state: DroneState, tag: str) -> bool:
    """
    Reads the drone's current GPS position and stores it as the launch
    lat/lon in DroneState. Call this once after connection, before takeoff.
    Returns True on success, False if GPS fix could not be obtained.
    """
    log.info(f"{tag} Reading launch position …")
    pos = get_gps(mav, timeout=5.0)
    if pos is None:
        log.error(f"{tag} Could not read launch position — no GPS fix")
        return False
    state.launch_lat, state.launch_lon = pos[0], pos[1]
    state.set_pos(*pos)   # also initialises the live telemetry
    log.info(f"{tag} Launch position: lat={pos[0]:.7f}  lon={pos[1]:.7f}  alt={pos[2]:.2f}m")
    return True

def run_planner(launch_positions: dict):
    """
    Pass launch positions directly into main_new and run it.
    launch_positions: {drone_index (int): (lat, lon)}
    """
    log.info("═" * 62)
    log.info("Running mission planner (main_new.py) …")
    log.info("═" * 62)
    main_height_diff.LAUNCH_LATLON   = launch_positions[1]
    main_height_diff.LAUNCH_POSITIONS = launch_positions
    main_height_diff.main()
    log.info("Mission planner finished.")


def load_missions() -> dict:
    """
    Read per-drone .waypoints files generated by run_planner.
    Returns {drone_index (int): [(lat, lon, alt), ...]}
    """
    pattern = os.path.join(MISSIONS_DIR, 'drone_*.waypoints')
    files   = sorted(glob.glob(pattern))
    if not files:
        log.error(f"No waypoint files found in {MISSIONS_DIR}/")
        return {}
    missions = {}
    for fp in files:
        try:
            idx = int(os.path.basename(fp)
                      .replace('drone_', '').replace('.waypoints', ''))
        except ValueError:
            continue
        wps = parse_waypoint_file(fp)
        if not wps:
            log.warning(f"D{idx}: empty waypoint file — skipping")
            continue
        missions[idx] = wps
        log.info(f"  D{idx}: {len(wps)} waypoints  ← {fp}")
    return missions

# ═══════════════════════════════════════════════════════════════════════════
#  PER-DRONE THREAD
# ═══════════════════════════════════════════════════════════════════════════

def drone_thread(state: DroneState, all_states: List[DroneState],gps_barrier: threading.Barrier, rtl_barrier: threading.Barrier,
                 skip_prearm: bool = False):
    tag = f"[D{state.index}|sys={state.sysid}]"

    # ── 1. Connect ───────────────────────────────────────────────────────────
    try:
        log.info(f"{tag} Connecting : {state.conn_str}")
        mav = mavutil.mavlink_connection(
            state.conn_str,
            source_system=255,       # GCS sysid (must differ from all drones)
            source_component=0,
            dialect='ardupilotmega',
        )
        wait_heartbeat(mav)
        # Lock onto this drone's sysid so messages to other drones are ignored
        mav.target_system    = state.sysid
        mav.target_component = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        request_streams(mav)
        log.info(f"{tag} Connected")

        # ── 2. Read launch position — retry until fix acquired ───────────────
        while True:
            if read_launch_position(mav, state, tag):
                break
            log.warning(f"{tag} No GPS fix yet — retrying in 2s …")
            time.sleep(2.0)
        
    except Exception as exc:
        state.error = f"Connection failed: {exc}"
        log.error(f"{tag} {state.error}")
        return
    
    # ── 3. Wait for all drones to read GPS, then run planner ─────────────────
    try:
        gps_barrier.wait()   # wait for all drones to reach this point
    except threading.BrokenBarrierError:
        log.error(f"{tag} Barrier broken — another drone failed to connect")
        return

    # Only drone 1 runs the planner; others wait at second barrier
    if state.index == 1:
        log.info(f"{tag} All drones have GPS fix — running mission planner …")
        run_planner({s.index: (s.launch_lat, s.launch_lon) for s in all_states})
        log.info(f"{tag} Mission planner complete !")

    try:
        gps_barrier.wait()   # wait for planner to finish before any drone proceeds
    except threading.BrokenBarrierError:
        log.error(f"{tag} Barrier broken — aborting")
        return
    
    # ── 4. Load mission waypoints generated by planner ───────────────────────
    missions = load_missions()
    if state.index not in missions:
        state.error = f"No mission file found for D{state.index}"
        log.error(f"{tag} {state.error}")
        return
    wps = missions[state.index]
    state.mission_wps = wps
    state.lawnmower_start = (wps[0][0], wps[0][1])
    log.info(f"{tag} Loaded {len(wps)} waypoints")

    # ── 5. Pre-arm checks ────────────────────────────────────────────────────
    if skip_prearm:
        log.warning(f"{tag} Pre-arm checks SKIPPED (--skip-prearm)")
    else:
        """log.info(f"{tag} Waiting for pre-arm checks …")
        t0 = time.time()
        while True:
            msg = mav.recv_match(type='SYS_STATUS', blocking=True, timeout=2)
            if msg is not None:
                present = msg.onboard_control_sensors_present
                healthy = msg.onboard_control_sensors_health
                if (present & healthy) == present:
                    log.info(f"{tag} Pre-arm checks passed ✓")
                    break
            if time.time() - t0 > ARM_TIMEOUT:
                state.error = "Pre-arm checks never passed"
                log.error(f"{tag} {state.error}")
                return
            time.sleep(1)"""
        #need to write pre-arm checks for 1. GPS Lock 2. Barometer 3. RC Channels 4.

    # ── 6. GUIDED mode + arm ─────────────────────────────────────────────────
    set_mode(mav, 4)
    time.sleep(0.5)

    log.info(f"{tag} Arming …")
    send_command_long(mav, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                      p1=1, p2=21196 if skip_prearm else 0)

    t0 = time.time()
    while True:
        if is_armed(mav):
            log.info(f"{tag} Armed !")
            break
        else:
            log.info(f"{tag} Waiting for Arm !")
            time.sleep(0.5)
            continue
    

    # ── 7. Takeoff ───────────────────────────────────────────────────────────
    log.info(f"{tag} Taking off to {TAKEOFF_ALT}m …")
    send_command_long(
        mav, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        p7=TAKEOFF_ALT,
    )

    t0 = time.time()
    while True:
        pos = get_gps(mav, timeout=2)
        if pos:
            state.set_pos(*pos)
            if pos[2] >= TAKEOFF_ALT * 0.92:
                log.info(f"{tag} Reached {pos[2]:.1f}m !")
                break
        time.sleep(1.0 / POLL_HZ)

    # ── 8. Fly waypoints one-by-one in GUIDED ────────────────────────────────
    log.info(f"{tag} Starting GUIDED waypoint execution …")
    set_mode(mav, 4)  # GUIDED
    curr_idx = state.index - 1

    for wp_idx, (wp_lat, wp_lon, wp_alt) in enumerate(wps):
        log.info(f"{tag} Flying to WP{wp_idx+1}/{len(wps)}: "
                 f"({wp_lat:.6f}, {wp_lon:.6f}, {wp_alt:.1f}m)")

        go_to(mav, wp_lat, wp_lon, wp_alt)

        while True:
            pos = get_gps(mav, timeout=1)
            if pos:
                state.set_pos(*pos)
                cur_lat, cur_lon, cur_alt = pos

                # ── Collision check ───────────────────────────────────────
                result = collision_check(all_states, curr_idx)

                if result is not None:
                    _, other_idx, lift_idx = result

                    if curr_idx == lift_idx:
                        # ── LIFT drone ────────────────────────────────────
                        log.warning(f"{tag} COLLISION RISK — LIFT drone, climbing")

                        # 1. Hold current position
                        hold_position(mav, cur_lat, cur_lon, cur_alt)

                        # 2. Climb until MIN_ALTI_DIST is met
                        lift_alt = cur_alt + LIFT
                        go_to(mav, cur_lat, cur_lon, lift_alt)
                        while True:
                            time.sleep(AVOIDANCE_HOLD_S)
                            pos2 = get_gps(mav, timeout=1)
                            if pos2:
                                state.set_pos(*pos2)
                                cur_lat, cur_lon, cur_alt = pos2
                            other_alt = all_states[other_idx].get_pos()[2]
                            if abs(cur_alt - other_alt) >= 0.95*LIFT:
                                log.info(f"{tag} Alt separation achieved "
                                         f"(alti_diff={abs(cur_alt - other_alt):.2f}m) ")
                                break
                            go_to(mav, cur_lat, cur_lon, lift_alt)

                        # 3. Hold at lifted alt while stay drone clears below
                        log.info(f"{tag} Holding at {lift_alt:.1f}m — "
                                 f"waiting for lower drone to clear")
                        while True:
                            time.sleep(AVOIDANCE_HOLD_S)
                            pos2 = get_gps(mav, timeout=1)
                            if pos2:
                                state.set_pos(*pos2)
                                cur_lat, cur_lon, cur_alt = pos2
                            hold_position(mav, cur_lat, cur_lon, cur_alt)
                            if collision_check(all_states, curr_idx) is None:
                                log.info(f"{tag} Lower drone cleared — resuming")
                                break

                        # 4. Descend to wp_alt and resume toward waypoint
                        go_to(mav, cur_lat, cur_lon, wp_alt)
                        log.info(f"{tag} Descending to {wp_alt:.1f}m, resuming WP{wp_idx+1}")
                        time.sleep(2.0)
                        go_to(mav, wp_lat, wp_lon, wp_alt)

                    else:
                        # ── STAY drone ────────────────────────────────────
                        log.warning(f"{tag} COLLISION RISK — STAY drone, "
                                    f"holding while lift drone climbs")

                        # Hold until lift drone has cleared
                        hold_position(mav, cur_lat, cur_lon, cur_alt)
                        while alti_check(all_states,curr_idx) <= 0.95*LIFT:
                            time.sleep(AVOIDANCE_HOLD_S)
                            pos2 = get_gps(mav, timeout=1)
                            if pos2:
                                state.set_pos(*pos2)
                                cur_lat, cur_lon, cur_alt = pos2
                            hold_position(mav, cur_lat, cur_lon, cur_alt)
                            if collision_check(all_states, curr_idx) is None:
                                log.info(f"{tag} Lift drone cleared — resuming")

                        # Resume toward current waypoint
                        go_to(mav, wp_lat, wp_lon, wp_alt)

            # Check if waypoint is reached
            if haversine(cur_lat, cur_lon, wp_lat, wp_lon) < 2.0 and \
               abs(cur_alt - wp_alt) < 1.0:
                log.info(f"{tag} WP{wp_idx+1} reached !")
                break

            time.sleep(1.0 / POLL_HZ)

    state.done = True
    log.info(f"{tag} All waypoints complete !")

    # ── 9. RTL in sequence — drones land one at a time by index ─────────────
    # Each drone waits for all lower-indexed drones to finish RTL before starting
    for turn in range(1, len(all_states) + 1):

        if state.index == turn:
            # My turn — start RTL
            log.info(f"{tag} My turn to RTL (turn {turn}/{len(all_states)}) …")
            set_mode(mav, 6)  # 6 = RTL

            # Wait until landed and disarmed
            while True:
                pos = get_gps(mav, timeout=2)
                if pos:
                    state.set_pos(*pos)
                    if pos[2] < 0.3:
                        log.info(f"{tag} Landed !  alt={pos[2]:.2f}m")
                        break
                hb = mav.recv_match(type='HEARTBEAT', blocking=False, timeout=0.2)
                if hb and not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    log.info(f"{tag} Disarmed — landed.")
                    break
                time.sleep(1)

        # All drones wait here — next drone only starts RTL once this one is done
        try:
            rtl_barrier.wait()
        except threading.BrokenBarrierError:
            log.error(f"{tag} RTL barrier broken at turn {turn}")
            break

    mav.close()
    log.info(f"{tag} Done — connection closed")
   

def main():
    parser = argparse.ArgumentParser(description='Swarm Mission Planner')
    parser.add_argument(
        '--skip-prearm', action='store_true',
        help='Skip pre-arm sensor checks and force-arm each drone',
    )
    args = parser.parse_args()
    skip_prearm: bool = args.skip_prearm

    if skip_prearm:
        log.warning(" ! --skip-prearm active: pre-arm checks will be bypassed")

    # Step 1 — build DroneState objects
    log.info("═" * 62)
    log.info("STEP 1 — Configuring swarm …")
    log.info("═" * 62)
    states: List[DroneState] = []
    for (idx, conn_str, sysid) in DRONES:
        s = DroneState(
            index    = idx,
            sysid    = sysid,
            conn_str = conn_str,
        )
        states.append(s)
        log.info(f"  D{idx}  sysid={sysid}  {conn_str}")

    if not states:
        log.error("No drones configured — check DRONES list")
        sys.exit(1)

    # Step 2 — launch per-drone threads
    # Barrier is set to 2 * len(states):
    # first wait  → all drones have GPS fix
    # second wait → planner has finished, all drones can proceed
    gps_barrier = threading.Barrier(len(states))
    rtl_barrier = threading.Barrier(len(states))

    log.info("═" * 62)
    log.info(f"STEP 2 — Launching {len(states)} drone thread(s) …")
    log.info("═" * 62)

    threads = []
    for s in states:
        t = threading.Thread(
            target=drone_thread,
            args=(s, states, gps_barrier, rtl_barrier, skip_prearm),
            name=f"D{s.index}",
            daemon=True,
        )
        threads.append(t)
        t.start()
        time.sleep(0.3)

    for t in threads:
        t.join()

    # Summary
    log.info("═" * 62)
    log.info("SWARM COMPLETE")
    log.info("═" * 62)
    for s in states:
        if s.error:
            log.error(f"  D{s.index}  ERROR  {s.error}")
        else:
            log.info(f"  D{s.index}  OK")


if __name__ == '__main__':
    main()