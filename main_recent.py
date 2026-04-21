"""
Multi-Drone Area Coverage Planner
===================================
- Partitions geofence via Power Diagram (balanced by real lawnmower time)
- Generates lawnmower path per drone
- Detects transit conflicts → altitude separation
- Exports per-drone .waypoints files (QGC WPL 110)
- Saves plot as PNG (no GUI window)

Usage:  python multi_drone.py
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no GUI — saves to file only
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely import affinity

filepath = "test.txt"

N_DRONES       = 2
ALTITUDE       = 6.0
ALT_SEPARATION = 5.0
HFOV           = 87.0
BUFFER_WIDTH   = 2.0
CRUISE_SPEED   = 2.0
YAW_RATE       = 45.0
SETTLING_TIME  = 1.5
OUTPUT_DIR     = 'missions'


def parse_waypoints(file_path):
    geofence = []
    launch = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header (first line)
    for i, line in enumerate(lines[1:]):
        parts = line.strip().split('\t')

        if len(parts) < 11:
            continue  # skip malformed lines

        lat = float(parts[8])
        lon = float(parts[9])

        if i == 0:
            # First waypoint = launch point
            launch = (lat, lon)
        else:
            geofence.append((lat, lon))

    return geofence, launch
# ═══════════════════════════════════════════════════
#  COORDINATE MATH
# ═══════════════════════════════════════════════════
GEOFENCE_LATLON,LAUNCH_LATLON = parse_waypoints(filepath)

R_EARTH = 6_371_000.0

def ll2xy(lat, lon, rlat, rlon):
    c = math.cos(math.radians(rlat))
    return (math.radians(lon - rlon) * R_EARTH * c,
            math.radians(lat - rlat) * R_EARTH)

def xy2ll(x, y, rlat, rlon):
    c = math.cos(math.radians(rlat))
    return (rlat + math.degrees(y / R_EARTH),
            rlon + math.degrees(x / (R_EARTH * c)))

def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R_EARTH * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def longest_edge_angle(pts):
    best, ang = 0, 0
    for i in range(len(pts)):
        dx = pts[(i+1) % len(pts)][0] - pts[i][0]
        dy = pts[(i+1) % len(pts)][1] - pts[i][1]
        d2 = dx*dx + dy*dy
        if d2 > best:
            best, ang = d2, math.degrees(math.atan2(dy, dx))
    return ang

def generate_lawnmower(cell, spacing, buf, rlat, rlon):
    if cell.is_empty or cell.area < 1: return []
    inner = cell.buffer(-buf)
    if inner.is_empty or inner.area < 1: inner = cell
    if isinstance(inner, MultiPolygon): inner = max(inner.geoms, key=lambda g: g.area)
    coords = list(inner.exterior.coords)[:-1]
    if len(coords) < 3: return []

    sa = longest_edge_angle(coords)
    rot = affinity.rotate(inner, -sa, origin=(0, 0))
    bx, by, mx, my = rot.bounds
    path, y, rev = [], by + spacing/2, False

    while y < my:
        hit = LineString([(bx-1, y), (mx+1, y)]).intersection(rot)
        if not hit.is_empty:
            segs = []
            if hit.geom_type == 'LineString': segs.append(list(hit.coords))
            elif hit.geom_type == 'MultiLineString':
                for p in hit.geoms: segs.append(list(p.coords))
            if rev:
                segs.reverse()
                for s in segs: s.reverse()
            for s in segs: path.extend(s)
            rev = not rev
        y += spacing

    if not path: return []
    ca, sa2 = math.cos(math.radians(sa)), math.sin(math.radians(sa))
    return [xy2ll(x*ca - y*sa2, x*sa2 + y*ca, rlat, rlon) for x, y in path]



def orient_paths_collision_safe(paths, cells, launch_ll, rlat, rlon):
    n = len(paths)
    if n < 2:
        for i in range(n):
            if len(paths[i]) >= 2:
                if haversine(launch_ll[0], launch_ll[1], paths[i][-1][0], paths[i][-1][1]) < \
                   haversine(launch_ll[0], launch_ll[1], paths[i][0][0], paths[i][0][1]):
                    paths[i].reverse()
        return paths

    centroids = [np.array([c.centroid.x, c.centroid.y]) if not c.is_empty
                 else np.array([0.,0.]) for c in cells]
    order = np.argsort([math.atan2(c[1], c[0]) for c in centroids])
    prev = None

    for rank, i in enumerate(order):
        if len(paths[i]) < 2: continue
        s, e = paths[i][0], paths[i][-1]
        if rank == 0:
            if haversine(launch_ll[0], launch_ll[1], e[0], e[1]) < \
               haversine(launch_ll[0], launch_ll[1], s[0], s[1]):
                paths[i].reverse()
            prev = paths[i][0]
        else:
            if haversine(prev[0], prev[1], s[0], s[1]) < \
               haversine(prev[0], prev[1], e[0], e[1]):
                paths[i].reverse()
            prev = paths[i][0]
    return paths


def plan_transit_altitudes(paths, cells, rlat, rlon):
    n = len(paths)
    launch = np.array([0., 0.])
    lines = []
    for i in range(n):
        if paths[i]:
            sx, sy = ll2xy(paths[i][0][0], paths[i][0][1], rlat, rlon)
            lines.append(LineString([launch, (sx, sy)]))
        else:
            lines.append(None)

    conflicts = [[] for _ in range(n)]
    for i in range(n):
        if not lines[i]: continue
        for j in range(n):
            if i == j or cells[j].is_empty: continue
            if lines[i].intersects(cells[j]):
                ix = lines[i].intersection(cells[j])
                if ix.length > 2.0:
                    conflicts[i].append(j)

    plans, layer = [], 0
    for i in range(n):
        if conflicts[i]:
            layer += 1
            plans.append({'transit_alt': ALTITUDE + layer * ALT_SEPARATION,
                          'coverage_alt': ALTITUDE, 'conflicts': conflicts[i],
                          'needs_climb': True})
        else:
            plans.append({'transit_alt': ALTITUDE, 'coverage_alt': ALTITUDE,
                          'conflicts': [], 'needs_climb': False})
    return plans

def build_mission(path_gps, ap):
    if not path_gps: return []
    m = []
    if ap['needs_climb']:
        m.append((path_gps[0][0], path_gps[0][1], ap['transit_alt']))
        m.append((path_gps[0][0], path_gps[0][1], ap['coverage_alt']))
        for lat, lon in path_gps[1:]:
            m.append((lat, lon, ap['coverage_alt']))
    else:
        for lat, lon in path_gps:
            m.append((lat, lon, ap['coverage_alt']))
    return m



def analyze_mission(path):
    if len(path) < 2: return 0., 0.
    t_trans, t_turn, t_yaw, prev_h = 0, 0, 0, None
    for i in range(len(path)-1):
        d = haversine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        h = math.degrees(math.atan2(path[i+1][1]-path[i][1], path[i+1][0]-path[i][0]))
        if prev_h is not None:
            t_yaw += abs((h - prev_h + 180) % 360 - 180) / YAW_RATE
        prev_h = h
        if i % 2 == 0: t_trans += d
        else: t_turn += d
    t = t_trans/CRUISE_SPEED + t_turn/(CRUISE_SPEED*0.6) + t_yaw + len(path)*SETTLING_TIME
    return t, t_trans + t_turn



def clip_half(poly, pi, pj, wi, wj):
    d = pj - pi; dist = np.linalg.norm(d)
    if dist < 1e-10: return poly
    n = d/dist
    mid = (pi+pj)*0.5 + n*((wi-wj)/(2.*dist))
    perp = np.array([-n[1], n[0]]); far = dist*12
    clip = Polygon([mid+perp*far, mid-perp*far, mid-perp*far-n*far, mid+perp*far-n*far])
    try:
        r = poly.intersection(clip)
        if isinstance(r, MultiPolygon): r = max(r.geoms, key=lambda g: g.area)
        return r if isinstance(r, Polygon) and not r.is_empty else Polygon()
    except: return Polygon()

def power_cells(sites, w, fence):
    cells = []
    for i in range(len(sites)):
        cell = fence
        for j in range(len(sites)):
            if i != j:
                cell = clip_half(cell, sites[i], sites[j], w[i], w[j])
                if cell.is_empty: break
        cells.append(cell)
    return cells

def seed_points(n, fence):
    bx, by, mx, my = fence.bounds
    step = max(mx-bx, my-by) / max(int(np.sqrt(n*8)), 4)
    pts = []
    x = bx + step*0.5
    while x < mx:
        y = by + step*0.5
        while y < my:
            if fence.contains(Point(x, y)): pts.append([x, y])
            y += step
        x += step
    pts = np.array(pts) if pts else np.array([[fence.centroid.x, fence.centroid.y]])
    if len(pts) <= n: return [tuple(p) for p in pts]
    sel = [0]
    for _ in range(n-1):
        d = np.min([np.linalg.norm(pts - pts[s], axis=1) for s in sel], axis=0)
        sel.append(int(np.argmax(d)))
    return [tuple(pts[s]) for s in sel]

def balance(seeds, fence, rlat, rlon, spacing, buf, iters=80, tol=0.03):
    n = len(seeds); w = np.zeros(n); sites = np.array(seeds)
    dd = [np.linalg.norm(sites[i]-sites[j]) for i in range(n) for j in range(i+1,n)]
    md = np.mean(dd) if dd else 100.
    best, bd, bw, bp, bt, bi = None, np.inf, w.copy(), [], np.zeros(n), []

    for it in range(iters):
        cells = power_cells(sites, w, fence)
        ok, times, paths = True, np.zeros(n), []
        for i in range(n):
            if cells[i].is_empty or cells[i].area < 1:
                ok = False; paths.append([]); continue
            paths.append(generate_lawnmower(cells[i], spacing, buf, rlat, rlon))
        if not ok: w *= 0.6; w -= w.mean(); continue

        paths = orient_paths_collision_safe(paths, cells, LAUNCH_LATLON, rlat, rlon)
        infos = []
        for i in range(n):
            tc, dc = analyze_mission(paths[i])
            tt = haversine(LAUNCH_LATLON[0], LAUNCH_LATLON[1],
                           paths[i][0][0], paths[i][0][1]) / CRUISE_SPEED if paths[i] else 0
            times[i] = tt + tc
            infos.append({'t_travel': tt, 't_coverage': tc, 'dist': dc,
                          'n_wp': len(paths[i]), 'area': cells[i].area})

        avg = times.mean()
        dev = np.abs(times - avg).max() / avg if avg > 0 else 0
        if dev < bd:
            bd, best, bw, bp, bt, bi = dev, list(cells), w.copy(), paths, times.copy(), infos
        if dev < tol: break
        g = times - avg; gm = np.abs(g).max()
        if gm > 1e-10: w -= md**2 * 0.08 * (0.95**it) * (g/gm)
        w -= w.mean()

    return best or cells, bw, bd, bp, bt, bi

# ═══════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════

COL  = ['#1D9E75','#534AB7','#D85A30','#D4537E','#378ADD','#639922']
FILL = ['#E1F5EE','#EEEDFE','#FAECE7','#FBEAF0','#E6F1FB','#EAF3DE']

def plot_mission(cells, paths, infos, times, fence, rlat, rlon, alt_plans, save_path):
    n = len(cells)
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_axes([0.04, 0.06, 0.48, 0.86])
    gx, gy = fence.exterior.xy
    ax.plot(gx, gy, 'k-', lw=2.2)
    ax.plot(0, 0, 's', color='black', ms=12, mec='white', mew=1.5, zorder=10)
    ax.annotate('LAUNCH', (0,0), fontsize=7, fontweight='bold', ha='center',
                va='top', xytext=(0,-10), textcoords='offset points', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85))

    for i in range(n):
        if cells[i].is_empty: continue
        c, f = COL[i%6], FILL[i%6]
        cx, cy = cells[i].exterior.xy
        ax.fill(cx, cy, alpha=0.2, color=f, ec=c, lw=1.5)
        if paths[i]:
            pxs = [ll2xy(la, lo, rlat, rlon)[0] for la, lo in paths[i]]
            pys = [ll2xy(la, lo, rlat, rlon)[1] for la, lo in paths[i]]
            ax.plot(pxs, pys, '-', color=c, lw=0.6, alpha=0.7)
            ap = alt_plans[i]
            lbl = f'D{i+1}\nSTART'
            if ap['needs_climb']: lbl += f'\n{ap["transit_alt"]:.0f}m→{ap["coverage_alt"]:.0f}m'
            ax.plot(pxs[0], pys[0], 'o', color=c, ms=9, mec='white', mew=1.5, zorder=8)
            ax.annotate(lbl, (pxs[0],pys[0]), fontsize=6, fontweight='bold', color=c,
                        ha='center', va='bottom', xytext=(0,8), textcoords='offset points', zorder=9)
            ax.plot(pxs[-1], pys[-1], 'X', color=c, ms=8, mew=2, zorder=8)
            ax.annotate('END', (pxs[-1],pys[-1]), fontsize=5.5, color=c, ha='center',
                        va='bottom', xytext=(0,6), textcoords='offset points', zorder=9)
            ls = '--' if ap['needs_climb'] else '-'
            ax.annotate('', xy=(pxs[0],pys[0]), xytext=(0,0),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.8, ls=ls, alpha=0.6))
            if ap['needs_climb']:
                ax.text(pxs[0]/2, pys[0]/2, f'{ap["transit_alt"]:.0f}m', fontsize=6.5,
                        color=c, ha='center', va='bottom', fontweight='bold', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='yellow', ec=c, alpha=0.6, lw=0.5))
            mid = len(pxs)//2
            if mid+1 < len(pxs):
                ax.annotate('', xy=(pxs[mid+1],pys[mid+1]), xytext=(pxs[mid],pys[mid]),
                            arrowprops=dict(arrowstyle='->', color=c, lw=2.5, alpha=0.4))
        cent = cells[i].centroid
        info = infos[i] if i < len(infos) else {}
        ax.text(cent.x, cent.y, f'D{i+1}\n{info.get("area",0):.0f}m²\n{info.get("n_wp",0)} wps',
                fontsize=7, ha='center', va='center', color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.75, lw=0.5))

    ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    ax.set_title(f'Multi-drone coverage — {n} drones @ {CRUISE_SPEED} m/s', fontsize=13, fontweight='bold')
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')

    ax2 = fig.add_axes([0.56, 0.5, 0.42, 0.42]); ax2.axis('off')
    hdr = f'{"Drone":>6}{"Area":>8}{"WPs":>6}{"Travel":>8}{"Cover":>8}{"Total":>8}{"Dist":>8}\n{"─"*56}\n'
    body = ''.join(f'  D{i+1:<4}{infos[i].get("area",0):>7.0f}{infos[i].get("n_wp",0):>6}'
                   f'{infos[i].get("t_travel",0):>7.1f}s{infos[i].get("t_coverage",0):>7.1f}s'
                   f'{times[i]:>7.1f}s{infos[i].get("dist",0):>7.0f}m\n' for i in range(n))
    tots = [t for t in times if t > 0]
    foot = (f'\n{"─"*56}\n  Max: {max(tots):.1f}s ({max(tots)/60:.1f}min)\n'
            f'  Min: {min(tots):.1f}s ({min(tots)/60:.1f}min)\n'
            f'  Dev: {max(tots)-min(tots):.1f}s ({(max(tots)-min(tots))/np.mean(tots)*100:.1f}%)\n'
            f'\n  Speed: {CRUISE_SPEED} m/s | Alt: {ALTITUDE}m\n'
            f'  HFOV: {HFOV}° | Buffer: {BUFFER_WIDTH}m\n'
            f'  Area: {fence.area:.0f}m² ({fence.area/10000:.2f}ha)\n') if tots else ''
    ax2.text(0, 1, hdr+body+foot, transform=ax2.transAxes, fontsize=9,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f8f8f6', ec='#ccc', alpha=0.95))

    if tots:
        ax3 = fig.add_axes([0.58, 0.06, 0.38, 0.36])
        y = np.arange(n)
        tr = [infos[i].get('t_travel',0) for i in range(n)]
        co = [infos[i].get('t_coverage',0) for i in range(n)]
        cs = [COL[i%6] for i in range(n)]
        ax3.barh(y, tr, height=0.5, color=[c+'55' for c in cs], ec=cs, lw=0.8, label='Travel')
        ax3.barh(y, co, height=0.5, left=tr, color=cs, alpha=0.5, ec=cs, lw=0.8, label='Coverage')
        ax3.axvline(np.mean(tots), color='gray', lw=0.8, ls=':', alpha=0.6, label='Avg')
        ax3.set_yticks(y); ax3.set_yticklabels([f'D{i+1}' for i in range(n)], fontweight='bold')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Mission time (real lawnmower estimate)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8, loc='lower right'); ax3.grid(axis='x', alpha=0.15)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Plot → {save_path}')



def export_missions(missions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exported = []
    for i, m in enumerate(missions):
        if not m: continue
        fp = os.path.join(out_dir, f'drone_{i+1}.waypoints')
        with open(fp, 'w') as f:
            f.write('QGC WPL 110\n')
            f.write(f'0\t1\t0\t16\t0.000000\t0.000000\t0.000000\t0.000000\t'
                    f'{LAUNCH_LATLON[0]:.6f}\t{LAUNCH_LATLON[1]:.6f}\t'
                    f'{ALTITUDE:.6f}\t1\n')
            for j, (lat, lon, alt) in enumerate(m):
                f.write(f'{j+1}\t0\t3\t16\t0.000000\t0.000000\t0.000000\t0.000000\t'
                        f'{lat:.6f}\t{lon:.6f}\t{alt:.6f}\t1\n')
        a0, a1 = m[0][2], m[-1][2]
        alt_s = f'{a0:.0f}→{a1:.0f}m' if a0 != a1 else f'{a1:.0f}m'
        print(f'    {fp}  ({len(m)} wps, alt={alt_s})')
        exported.append(fp)
    return exported


def main():
   
    print(f'  MULTI-DRONE COVERAGE PLANNER')
    print(f'  {N_DRONES} drones | {CRUISE_SPEED} m/s | alt={ALTITUDE}m')


    rlat, rlon = LAUNCH_LATLON
    local = [ll2xy(la, lo, rlat, rlon) for la, lo in GEOFENCE_LATLON]
    fence = Polygon(local)
    if not fence.is_valid: fence = fence.buffer(0)

    spacing = 2 * ALTITUDE * math.tan(math.radians(HFOV / 2))
    print(f'\n  Geofence: {fence.area:.0f}m²')
    print(f'  Sweep spacing: {spacing:.1f}m')

    seeds = seed_points(N_DRONES, fence)
    
    cells, wts, dev, paths, times, infos = balance(seeds, fence, rlat, rlon, spacing, BUFFER_WIDTH)
    print(f'  Deviation: {dev*100:.1f}%')

    print(f'\n  Transit altitude check:')
    alt_plans = plan_transit_altitudes(paths, cells, rlat, rlon)
    missions = []
    for i in range(N_DRONES):
        missions.append(build_mission(paths[i], alt_plans[i]))
        c = alt_plans[i]['conflicts']
        if c: print(f'    D{i+1}: crosses D{[x+1 for x in c]} → transit {alt_plans[i]["transit_alt"]:.0f}m, descend to {ALTITUDE:.0f}m')
        else: print(f'    D{i+1}: clear → {ALTITUDE:.0f}m')

    print(f'\n  {"─"*55}')
    for i in range(N_DRONES):
        info, ap = infos[i], alt_plans[i]
        a = f'{ap["transit_alt"]:.0f}→{ap["coverage_alt"]:.0f}m' if ap['needs_climb'] else f'{ap["coverage_alt"]:.0f}m'
        print(f'  D{i+1}: {info["area"]:.0f}m² | {info["n_wp"]} wps | '
              f'travel={info["t_travel"]:.1f}s cover={info["t_coverage"]:.1f}s '
              f'total={times[i]:.1f}s | alt={a}')
    print(f'  {"─"*55}')
    tots = [t for t in times if t > 0]
    if tots:
        print(f'  Range: {min(tots):.1f}–{max(tots):.1f}s '
              f'(Δ{max(tots)-min(tots):.1f}s / {(max(tots)-min(tots))/np.mean(tots)*100:.1f}%)')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_mission(cells, paths, infos, times, fence, rlat, rlon,
                 alt_plans, os.path.join(OUTPUT_DIR, 'mission_plan.png'))

    print(f'\n  Exporting to {OUTPUT_DIR}/:')
    exported = export_missions(missions, OUTPUT_DIR)
    print(f'\n  Done. {len(exported)} files exported to {OUTPUT_DIR}/')

if __name__ == '__main__':
    main()