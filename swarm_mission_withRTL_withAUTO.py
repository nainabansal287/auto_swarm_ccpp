#changes:
#1. If 1st drone flies at 6m, next at 8m, 10m so on.
#2. Drones will perform mode RTL in sequence instead of LAND at final waypoint.
#3. After takeoff, mission is UPLOADED and executed in AUTO mode (no more
#   per-waypoint GUIDED go_to). Collision avoidance briefly switches to
#   GUIDED to lift/hold, then returns to AUTO and the mission resumes from
#   the current MISSION_CURRENT.seq.

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
MISSIONS_DIR = '/home/naina/Downloads/drone_swarm/mission'
#DRONE_DELAY  = 60.0    # seconds between consecutive takeoffs - NOW REMOVED
TAKEOFF_ALT  = 6.0     # metres (relative / AGL)
MIN_HORIZ_DIST   = 2.0   # metres — trigger hold if closer than this
#MIN_ALTI_DIST = 2.0
AVOIDANCE_HOLD_S = 2.0   # seconds between re-checks while holding
LIFT             = 2.0   # alti increase in case of collision

CONNECT_TIMEOUT  = 30   # seconds — wait for first heartbeat
#ARM_TIMEOUT      = 30    # seconds
POLL_HZ = 4

# WP-reach thresholds (used only for the "last WP" position fallback)
WP_HORIZ_TOL = 2.0     # metres
WP_VERT_TOL  = 1.0     # metres

#Drone connection strings
DRONES: List[Tuple[int, str, int]] = [
    # (1-based index,  conn_str, sysid)
    (1,  'udpin:0.0.0.0:14555',  1),
    (2,  'udpin:0.0.0.0:14565',  2),
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

def set_mode(mav, mode_id: int, poll_s: float = 0.4,
             timeout: Optional[float] = None) -> bool:
    """
    Switch flight mode. Re-sends set_mode_send and polls HEARTBEAT every
    `poll_s` seconds until the mode is confirmed. Loops indefinitely by
    default — pass `timeout` (seconds) for a hard cap.

    Bails out (returns False) if the drone disarms mid-attempt, so we don't
    spin forever after a DISARM_DELAY auto-disarm.
    """
    modes = { #note: this only applies to ardupilot, will have to change w px4
        0: 'STABILIZE',
        3: 'AUTO',
        4: 'GUIDED',
        5: 'LOITER',
        6: 'RTL',
        9: 'LAND'
    }
    target_name = modes.get(mode_id, str(mode_id))
    t0 = time.time()
    attempts = 0

    while True:
        # Spam the request — single UDP packets get dropped silently
        mav.mav.set_mode_send(
            mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        attempts += 1

        hb = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=poll_s)
        if hb is not None:
            if hb.custom_mode == mode_id:
                log.info(f"    Mode → {target_name}  "
                         f"(took {attempts} attempt(s), {time.time()-t0:.1f}s)")
                return True
            # If we got disarmed while trying, stop hammering
            if not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                # Tolerate being disarmed at the *start* (e.g. RTL is allowed
                # to be set while disarmed). Only bail if we'd been armed and
                # got disarmed mid-attempt.
                if attempts > 2:
                    log.error(f"    Drone disarmed while waiting for "
                              f"{target_name} — bailing")
                    return False

        # Hard timeout if caller asked for one
        if timeout is not None and (time.time() - t0) >= timeout:
            log.warning(f"    set_mode({target_name}) — no confirmation "
                        f"within {timeout}s after {attempts} attempts")
            return False

        # Periodic operator notice every ~4s so the log isn't silent
        if attempts % 10 == 0:
            log.warning(f"    Still trying to set {target_name}…  "
                        f"(attempts={attempts}, elapsed={time.time()-t0:.1f}s)")

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

    # Clear any existing mission first to avoid stale items
    mav.mav.mission_clear_all_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    # Drain any straggling ack from the clear
    mav.recv_match(type='MISSION_ACK', blocking=True, timeout=2)

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
            if time.time() - t0 > 20:
                raise RuntimeError(f"{tag} mission upload timed out waiting for MISSION_REQUEST")
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

def alti_check(states: List[DroneState], curr_idx: int) -> Optional[float]:
    """
    Returns vertical separation (m) from the closest in-transit drone within
    MIN_HORIZ_DIST horizontally. None if no such drone exists.
    """
    current = states[curr_idx]
    curr_lat, curr_lon, curr_alt = current.get_pos()

    for i, other in enumerate(states):
        if i == curr_idx or not other.in_transit:
            continue
        o_lat, o_lon, o_alt = other.get_pos()
        d = haversine(curr_lat, curr_lon, o_lat, o_lon)
        if d < MIN_HORIZ_DIST:
            return abs(curr_alt - o_alt)
    return None


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

def drone_thread(state: DroneState, all_states: List[DroneState],
                 gps_barrier: threading.Barrier, rtl_barrier: threading.Barrier,
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

    # ── 6. Wait for GCS / RC to arm ──────────────────────────────────────────
    # NOTE: ArduCopter blocks RC-stick arming in GUIDED, so we let the operator
    # arm in their normal mode (LOITER / ALT_HOLD / STABILIZE) and switch to
    # GUIDED right after. The retrying set_mode below makes that switch reliable
    # against dropped UDP packets — DO NOT move GUIDED before the arm-wait.
    time.sleep(0.5)

    while True:
        if is_armed(mav):
            log.info(f"{tag} Armed !")
            break
        log.info(f"{tag} Waiting for GCS to arm...")
        time.sleep(0.5)

    # ── 7. Switch to GUIDED and take off ─────────────────────────────────────
    # DISARM_DELAY on Copter is ~10 s, so we must get into GUIDED + take off
    # promptly after arming. set_mode now re-sends every 1 s for up to 5 s.
    takeoff_alt = TAKEOFF_ALT + 2 * (state.index - 1)
    log.info(f"{tag} Switching to GUIDED for takeoff …")
    if not set_mode(mav, 4):
        state.error = "Could not switch to GUIDED after arming (disarmed?)"
        log.error(f"{tag} {state.error}")
        return

    log.info(f"{tag} Taking off to {takeoff_alt}m …")
    send_command_long(
        mav, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        p7=takeoff_alt,
    )

    t_takeoff = time.time()
    while True:
        pos = get_gps(mav, timeout=2)
        if pos:
            state.set_pos(*pos)
            if pos[2] >= takeoff_alt * 0.95:
                log.info(f"{tag} Reached {pos[2]:.1f}m !")
                break

        
        time.sleep(1.0 / POLL_HZ)

    # ── 8. Upload mission and execute in AUTO mode ───────────────────────────
    log.info(f"{tag} Uploading mission to autopilot …")
    try:
        upload_mission(mav, wps, tag)
    except Exception as exc:
        state.error = f"Mission upload failed: {exc}"
        log.error(f"{tag} {state.error}")
        return

    # Make sure the mission starts at the first real WP (seq=1), not home (seq=0)
    mav.mav.mission_set_current_send(
        mav.target_system, mav.target_component, 1
    )

    log.info(f"{tag} Switching to AUTO — autopilot will fly the mission …")
    if not set_mode(mav, 3):
        state.error = "Could not switch to AUTO (disarmed?)"
        log.error(f"{tag} {state.error}")
        return

    curr_idx     = state.index - 1
    last_wp_seq  = len(wps)        # WP seqs are 1..len(wps); seq=0 is home
    last_wp_lat, last_wp_lon, last_wp_alt = wps[-1]
    mission_complete = False
    highest_reached  = 0

    while not mission_complete:
        # ── Telemetry ────────────────────────────────────────────────────
        pos = get_gps(mav, timeout=1)
        if pos:
            state.set_pos(*pos)
            cur_lat, cur_lon, cur_alt = pos
        else:
            time.sleep(1.0 / POLL_HZ)
            continue

        # ── Mission progress (non-blocking peek) ─────────────────────────
        mir = mav.recv_match(type='MISSION_ITEM_REACHED', blocking=False)
        if mir is not None:
            highest_reached = max(highest_reached, mir.seq)
            log.info(f"{tag} MISSION_ITEM_REACHED  seq={mir.seq}/{last_wp_seq}")
            if mir.seq >= last_wp_seq:
                log.info(f"{tag} Final waypoint reached — mission complete !")
                mission_complete = True
                break

        # Position-based fallback in case MISSION_ITEM_REACHED was missed
        if (haversine(cur_lat, cur_lon, last_wp_lat, last_wp_lon) < WP_HORIZ_TOL
                and abs(cur_alt - last_wp_alt) < WP_VERT_TOL):
            mc = mav.recv_match(type='MISSION_CURRENT', blocking=False)
            if mc is not None and mc.seq >= last_wp_seq:
                log.info(f"{tag} Final waypoint reached (position fallback) !")
                mission_complete = True
                break

        # ── Collision avoidance — pause AUTO, do GUIDED maneuver ─────────
        result = collision_check(all_states, curr_idx)
        if result is not None:
            _, other_idx, lift_idx = result

            log.warning(f"{tag} COLLISION RISK — pausing AUTO, switching to GUIDED")
            set_mode(mav, 4)  # GUIDED

            if curr_idx == lift_idx:
                # ── LIFT drone ────────────────────────────────────────────
                log.warning(f"{tag} LIFT drone, climbing")

                # 1. Hold current position briefly
                hold_position(mav, cur_lat, cur_lon, cur_alt)

                # 2. Climb until separation is met
                lift_alt = cur_alt + LIFT
                go_to(mav, cur_lat, cur_lon, lift_alt)
                while True:
                    time.sleep(AVOIDANCE_HOLD_S)
                    pos2 = get_gps(mav, timeout=1)
                    if pos2:
                        state.set_pos(*pos2)
                        cur_lat, cur_lon, cur_alt = pos2
                    other_alt = all_states[other_idx].get_pos()[2]
                    if abs(cur_alt - other_alt) >= 0.95 * LIFT:
                        log.info(f"{tag} Alt separation achieved "
                                 f"(alti_diff={abs(cur_alt - other_alt):.2f}m)")
                        break
                    go_to(mav, cur_lat, cur_lon, lift_alt)

                # 3. Hold at lifted alt until stay drone clears below
                log.info(f"{tag} Holding at {lift_alt:.1f}m — waiting for lower drone to clear")
                while True:
                    time.sleep(AVOIDANCE_HOLD_S)
                    pos2 = get_gps(mav, timeout=1)
                    if pos2:
                        state.set_pos(*pos2)
                        cur_lat, cur_lon, cur_alt = pos2
                    hold_position(mav, cur_lat, cur_lon, cur_alt)
                    if collision_check(all_states, curr_idx) is None:
                        log.info(f"{tag} Lower drone cleared")
                        break

            else:
                # ── STAY drone ────────────────────────────────────────────
                log.warning(f"{tag} STAY drone, holding while lift drone climbs")
                hold_position(mav, cur_lat, cur_lon, cur_alt)
                while True:
                    time.sleep(AVOIDANCE_HOLD_S)
                    pos2 = get_gps(mav, timeout=1)
                    if pos2:
                        state.set_pos(*pos2)
                        cur_lat, cur_lon, cur_alt = pos2
                    hold_position(mav, cur_lat, cur_lon, cur_alt)

                    sep = alti_check(all_states, curr_idx)
                    if sep is None:
                        log.info(f"{tag} Lift drone cleared — resuming")
                        break
                    if sep >= 0.95 * LIFT:
                        log.info(f"{tag} Sufficient alt separation "
                                 f"(alti_diff={sep:.2f}m) — resuming")
                        break

            # Hand control back to the mission
            log.info(f"{tag} Resuming AUTO — mission continues from current seq")
            set_mode(mav, 3)  # AUTO

        time.sleep(1.0 / POLL_HZ)

    state.done = True
    log.info(f"{tag} All waypoints complete !")

    # ── 9. RTL in sequence — drones land one at a time by index ──────────────
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
