import json
import math
import random
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import astar_with_waypoints as astar_module
from astar_with_waypoints import astar, astar_with_waypoints, on_barrier

GRID_W = 50
GRID_H = 30


def in_bounds(x, y):
    return 0 <= x <= GRID_W and 0 <= y <= GRID_H


def random_point():
    return [random.randint(0, GRID_W), random.randint(0, GRID_H)]


def random_env(min_h=2, max_h=4, min_v=2, max_v=4):
    h_count = random.randint(min_h, max_h)
    v_count = random.randint(min_v, max_v)
    h_barriers = []
    h_min_span = max(4, round(4 * GRID_W / 50))
    h_max_span = max(h_min_span, round(18 * GRID_W / 50))
    for _ in range(h_count):
        y = random.randint(1, GRID_H - 1)
        span = random.randint(h_min_span, min(h_max_span, GRID_W))
        x1 = random.randint(0, max(0, GRID_W - span))
        x2 = x1 + span
        h_barriers.append([y, x1, x2])
    v_barriers = []
    v_min_span = max(4, round(4 * GRID_H / 30))
    v_max_span = max(v_min_span, round(18 * GRID_H / 30))
    for _ in range(v_count):
        x = random.randint(1, GRID_W - 1)
        span = random.randint(v_min_span, min(v_max_span, GRID_H))
        y1 = random.randint(0, max(0, GRID_H - span))
        y2 = y1 + span
        v_barriers.append([x, y1, y2])
    env = {"Horizontal Barriers": h_barriers, "Vertical Barriers": v_barriers}
    for _ in range(200):
        s = random_point()
        g = random_point()
        if s == g:
            continue
        if on_barrier(s[0], s[1], env) or on_barrier(g[0], g[1], env):
            continue
        p, _, _ = astar(tuple(s), tuple(g), env, GRID_W, GRID_H)
        if p is not None and len(p) >= 2:
            env["Start Point"] = s
            env["Goal Point"] = g
            return env
    return None


def random_waypoints(env, k_min=3, k_max=7):
    s = env["Start Point"]
    g = env["Goal Point"]
    k = random.randint(k_min, k_max)
    mids = k - 2
    wps = [list(s)]
    if mids <= 0:
        wps.append(list(g))
        return wps
    max_offset = max(9, round(9 * min(GRID_W / 50, GRID_H / 30)))
    added = 0
    for i in range(1, mids + 1):
        t = i / (mids + 1)
        base_x = int(round(s[0] + (g[0] - s[0]) * t))
        base_y = int(round(s[1] + (g[1] - s[1]) * t))
        p = None
        for r in range(0, max_offset + 1):
            for dx, dy in [
                (0, 0),
                (r, 0), (-r, 0), (0, r), (0, -r),
                (r, r), (r, -r), (-r, r), (-r, -r),
            ]:
                nx = base_x + dx
                ny = base_y + dy
                if not in_bounds(nx, ny):
                    continue
                if on_barrier(nx, ny, env):
                    continue
                p = [nx, ny]
                break
            if p is not None:
                break
        if p is None:
            p = [base_x, base_y] if in_bounds(base_x, base_y) and not on_barrier(base_x, base_y, env) else list(s)
        wps.append(p)
        added += 1
    wps.append(list(g))
    return wps


def compute_label(env, wps):
    s = env["Start Point"]
    g = env["Goal Point"]
    dp, d_op, d_st = astar(tuple(s), tuple(g), env, GRID_W, GRID_H)
    if dp is None or d_op <= 0 or d_st <= 0:
        return None
    wp, wp_op, wp_st = astar_with_waypoints(wps, env)
    if wp is None or len(wp) < 2:
        return None
    if wp_op <= 0:
        wp_op = d_op
    if wp_st <= 0:
        wp_st = d_st
    op_ratio = wp_op / d_op
    st_ratio = wp_st / d_st
    label = 1.0 - math.sqrt(op_ratio * st_ratio)
    return {
        "label": label,
        "baseline_op": d_op,
        "baseline_storage": d_st,
        "waypoint_op": wp_op,
        "waypoint_storage_peak": wp_st,
        "waypoints": wps,
        "path": wp,
        "env": env,
    }


def build_windows_sample(rec, window_len=2):
    wps = rec["waypoints"]
    if len(wps) < window_len + 1:
        return None
    window = []
    for i in range(0, window_len):
        a = wps[i]
        b = wps[i + 1]
        window.append([int(a[0]), int(a[1]), int(b[0]), int(b[1])])
    return {"window": window, "label": rec["label"]}


def compute_away(former, chosen, goal):
    h = math.hypot(former[0] - goal[0], former[1] - goal[1])
    if h < 1e-6:
        h = 1.0
    return math.hypot(former[0] - chosen[0], former[1] - chosen[1]) / h


def compute_prox(chosen, start, goal):
    dg = math.hypot(chosen[0] - goal[0], chosen[1] - goal[1])
    ds = math.hypot(chosen[0] - start[0], chosen[1] - start[1])
    base = ds + dg
    if base < 1e-6:
        return 0.0
    return 1.0 - dg / base


def build_neural_path_record(rec):
    path = rec["path"]
    if len(path) < 2:
        return None
    return {
        "path": [[int(p[0]), int(p[1])] for p in path],
        "label": rec["label"],
    }


def write_neural_module(records, out_path):
    paths = [r["path"] for r in records]
    scores = [round(float(r["label"]), 6) for r in records]
    lines = []
    lines.append("import math\n")
    lines.append("path_map=[")
    for i, p in enumerate(paths):
        inner = ", ".join("[" + str(pt[0]) + ", " + str(pt[1]) + "]" for pt in p)
        comma = "," if i < len(paths) - 1 else ""
        lines.append("    [" + inner + "]" + comma)
    lines.append("    ]")
    lines.append("score_map=[" + ", ".join(str(v) for v in scores) + "]\n")
    lines.append("def compute_away(former_point_x,former_point_y,chosen_point_x,chosen_point_y,goal_x,goal_y)->float:")
    lines.append("    heuristic = math.sqrt((former_point_x - goal_x) ** 2 + (former_point_y - goal_y) ** 2)")
    lines.append("    AwayFromFormer = math.sqrt((chosen_point_x - former_point_x) ** 2 + (chosen_point_y - former_point_y) ** 2) / heuristic if heuristic>1e-6 else 0.0")
    lines.append("    return AwayFromFormer\n")
    lines.append("def compute_prox(chosen_point_x,chosen_point_y,start_x,start_y,goal_x,goal_y)->float:")
    lines.append("    dg = math.sqrt((chosen_point_x - goal_x) ** 2 + (chosen_point_y - goal_y) ** 2)")
    lines.append("    ds = math.sqrt((chosen_point_x - start_x) ** 2 + (chosen_point_y - start_y) ** 2)")
    lines.append("    proximity_score = 1 - dg / (ds + dg) if ds+dg>1e-6 else 0.0")
    lines.append("    return proximity_score\n")
    lines.append("def train_dataset():")
    lines.append("    x_train=[]")
    lines.append("    y_train=[]")
    lines.append("    for j in range(len(path_map)):")
    lines.append("        path = path_map[j]")
    lines.append("        start_x = path[0][0]")
    lines.append("        start_y = path[0][1]")
    lines.append("        goal_x = path[-1][0]")
    lines.append("        goal_y = path[-1][1]")
    lines.append("        for i in range(1,len(path)):")
    lines.append("            former_point_x=path[i-1][0]")
    lines.append("            former_point_y=path[i-1][1]")
    lines.append("            chosen_point_x=path[i][0]")
    lines.append("            chosen_point_y=path[i][1]")
    lines.append("            away=compute_away(former_point_x,former_point_y,chosen_point_x,chosen_point_y,goal_x,goal_y)")
    lines.append("            prox=compute_prox(chosen_point_x,chosen_point_y,start_x,start_y,goal_x,goal_y)")
    lines.append("            x_train.append([away,prox])")
    lines.append("            y_train.append(score_map[j])")
    lines.append("    return x_train,y_train\n")
    lines.append("x_train,y_train=train_dataset()\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_count(n, builder, min_label=-5.0, max_label=2.0):
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 400:
        attempts += 1
        env = random_env()
        if env is None:
            continue
        wps = random_waypoints(env)
        rec = compute_label(env, wps)
        if rec is None:
            continue
        if not (min_label <= rec["label"] <= max_label):
            continue
        sample = builder(rec)
        if sample is None:
            continue
        out.append(sample)
    return out


def write_windows_module(samples, out_path):
    windows = []
    labels = []
    for s in samples:
        windows.append(s["window"])
        labels.append(round(float(s["label"]), 6))
    w_lines = []
    w_lines.append("import tensorflow as tf\n")
    w_lines.append("x_train=tf.constant([")
    for i, w in enumerate(windows):
        inner = ", ".join(
            "[" + ", ".join(str(v) for v in step) + "]" for step in w
        )
        comma = "," if i < len(windows) - 1 else ""
        w_lines.append("    [" + inner + "]" + comma)
    w_lines.append("])\n")
    w_lines.append("# label set: label=1-sqrt(waypoint_op/baseline_op * waypoint_storage_peak/baseline_storage)")
    y_str = ", ".join(str(v) for v in labels)
    w_lines.append("y_train=tf.constant([" + y_str + "])\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(w_lines))


def main():
    parser = argparse.ArgumentParser(description="Generate LSTM training data on a configurable grid.")
    parser.add_argument("--width", type=int, default=50, help="Maximum x coordinate.")
    parser.add_argument("--height", type=int, default=30, help="Maximum y coordinate.")
    parser.add_argument("--samples", type=int, default=100, help="Number of records per dataset.")
    parser.add_argument("--seed", type=int, default=20261006, help="Random seed.")
    parser.add_argument(
        "--suffix",
        default=None,
        help="Output filename suffix (defaults to the sample count).",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.samples < 1:
        parser.error("--width, --height, and --samples must all be positive.")

    global GRID_W, GRID_H
    GRID_W = args.width
    GRID_H = args.height
    random.seed(args.seed)
    astar_module.PATH = os.devnull

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "..", "resources")
    os.makedirs(res_dir, exist_ok=True)
    suffix = args.suffix or str(args.samples)
    windows_stem = f"windows_trainset_{suffix}"
    neural_stem = f"neural_trainset_{suffix}"

    win_samples = generate_count(args.samples, build_windows_sample)
    neu_records = generate_count(args.samples, build_neural_path_record)

    with open(os.path.join(res_dir, windows_stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(win_samples, f, indent=2)
    with open(os.path.join(res_dir, neural_stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(neu_records, f, indent=2)

    write_windows_module(
        win_samples,
        os.path.join(base_dir, windows_stem + ".py"),
    )
    write_neural_module(
        neu_records,
        os.path.join(base_dir, neural_stem + ".py"),
    )

    print(f"grid: {GRID_W}x{GRID_H}")
    print("windows samples:", len(win_samples), "->", windows_stem)
    print("neural  path records:", len(neu_records), "->", neural_stem)
    if win_samples:
        ls = [s["label"] for s in win_samples]
        print("windows label min/avg/max:", min(ls), sum(ls) / len(ls), max(ls))
    if neu_records:
        ls = [s["label"] for s in neu_records]
        print("neural  path label min/avg/max:", min(ls), sum(ls) / len(ls), max(ls))


if __name__ == "__main__":
    main()
