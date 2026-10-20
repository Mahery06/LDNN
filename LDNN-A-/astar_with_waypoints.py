import heapq
import math
import os
import json
import ast
import numpy as np
from matplotlib import pyplot as plt

NEIGHBORS = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

PATH = "result/hybrid2.txt"

def _is_collision(chosen_point_x, chosen_point_y, former_point_x, former_point_y, env: dict) -> int:
    vertical_barriers = env.get("Vertical Barriers", [])
    horizontal_barriers = env.get("Horizontal Barriers", [])
    tensor_test = np.array([[chosen_point_x - former_point_x], [chosen_point_y - former_point_y]])

    for vertical_barrier in vertical_barriers:
        if former_point_x == chosen_point_x == vertical_barrier[0]:
            if not (max(chosen_point_y, former_point_y) < vertical_barrier[1] or min(chosen_point_y, former_point_y) > vertical_barrier[2]):
                return 1
        end1 = [vertical_barrier[0], vertical_barrier[1]]
        end2 = [vertical_barrier[0], vertical_barrier[2]]

        tensor = np.array([[end1[0] - former_point_x, end2[0] - former_point_x], [end1[1] - former_point_y, end2[1] - former_point_y]])
        try:
            alpha, beta = np.linalg.solve(tensor, tensor_test)
            if alpha[0] >= 0 and beta[0] >= 0:
                if min(chosen_point_x, former_point_x) <= vertical_barrier[0] <= max(chosen_point_x, former_point_x):
                    return 1
        except np.linalg.LinAlgError:
            continue

    for horizontal_barrier in horizontal_barriers:
        if former_point_y == chosen_point_y == horizontal_barrier[0]:
            if not (max(chosen_point_x, former_point_x) < horizontal_barrier[1] or min(chosen_point_x, former_point_x) > horizontal_barrier[2]):
                return 1
        end1 = [horizontal_barrier[1], horizontal_barrier[0]]
        end2 = [horizontal_barrier[2], horizontal_barrier[0]]

        tensor = np.array([[end1[0] - former_point_x, end2[0] - former_point_x], [end1[1] - former_point_y, end2[1] - former_point_y]])
        try:
            alpha, beta = np.linalg.solve(tensor, tensor_test)
            if alpha[0] >= 0 and beta[0] >= 0:
                if min(chosen_point_y, former_point_y) <= horizontal_barrier[0] <= max(chosen_point_y, former_point_y):
                    return 1
        except np.linalg.LinAlgError:
            continue

    return 0


def on_barrier(x, y, env):
    for hb in env["Horizontal Barriers"]:
        if y == hb[0] and hb[1] <= x <= hb[2]:
            return True
    for vb in env["Vertical Barriers"]:
        if x == vb[0] and vb[1] <= y <= vb[2]:
            return True
    return False


def astar(start, goal, env, grid_w=50, grid_h=30, open_recorder=None):
    start = tuple(start)
    goal = tuple(goal)

    if on_barrier(goal[0], goal[1], env):
        return None, 0, 0

    def h(pos):
        return (pos[0] - goal[0])**2+ (pos[1] - goal[1])**2

    def in_bounds(x, y):
        return 0 <= x <= grid_w and 0 <= y <= grid_h

    operation_cost=0
    open_set = [(h(start),0,start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    pushed = 1

    while open_set:
        _,_,cur= heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        operation_cost += 1 # 由于每一次pop都会产生f-cost与neighbors的计算，operation cost只统计pop的次数

        if cur == goal:
            path = [list(cur)]
            while cur in came_from:
                cur = came_from[cur]
                path.append(list(cur))
            path.reverse()
            storage_cost = pushed
            return path, operation_cost, storage_cost

        for dx, dy in NEIGHBORS:
            nx, ny = cur[0]+dx, cur[1]+dy
            nb = (nx, ny)
            if not in_bounds(nx, ny) or nb in closed or on_barrier(nx, ny, env):
                continue

            gt = g_score[cur] + dx ** 2 + dy ** 2
            if nb not in g_score or gt < g_score[nb]:
                came_from[nb] = cur
                g_score[nb] = gt
                heapq.heappush(open_set, (gt + h(nb), gt,nb))
                pushed += 1
                if open_recorder is not None:
                    open_recorder.add(nb)



    storage_cost = pushed
    return None, operation_cost, storage_cost

def map_features(env, start, goal):

    hb=env["Horizontal Barriers"]
    vb=env["Vertical Barriers"]

    obstacles=0
    for h in hb:
        obstacles+=(h[2]-h[1]+1)
    for v in vb:
        obstacles+=(v[2]-v[1]+1)

    rho = obstacles / 1500
    D = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return rho, D


def adaptive_params(env, start, goal):
    """自适应计算 α、β"""
    RHO_REF = 0.15
    alpha0 = 1.5;beta0 = 1.0
    D_REF = 78
    rho, D = map_features(env, start, goal)
    alpha = alpha0 * (D / D_REF)
    beta  = beta0 * (rho / RHO_REF)
    alpha = max(1.0, min(2.0, alpha))
    beta  = max(0.75, min(1.0, beta))
    return alpha, beta

def llm_astar(start, goal, env, grid_w=50, grid_h=30, open_recorder=None):
    start = tuple(start)
    target = tuple(goal)
    goal = tuple(env["Goal Point"])

    alpha, beta = adaptive_params(env, start , goal)

    if on_barrier(target[0], target[1], env):
        return None, 0, 0

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + (math.sqrt(2)-2)*min(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))

    def cost_to_target(pos, target):
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def in_bounds(x, y):
        return 0 <= x <= grid_w and 0 <= y <= grid_h

    operation_cost = 0
    open_set = [(alpha * h(start) + beta*cost_to_target(start, target), 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    pushed = 1

    while open_set:
        _, _, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        operation_cost += 1  # 由于每一次pop都会产生f-cost与neighbors的计算，operation cost只统计pop的次数

        if cur == target:
            path = [list(cur)]
            while cur in came_from:
                cur = came_from[cur]
                path.append(list(cur))
            path.reverse()
            storage_cost = pushed
            return path, operation_cost, storage_cost

        for dx, dy in NEIGHBORS:
            nx, ny = cur[0] + dx, cur[1] + dy
            nb = (nx, ny)
            if not in_bounds(nx, ny) or nb in closed or on_barrier(nx, ny, env):
                continue

            gt = g_score[cur] + abs(dx) + abs(dy)
            if nb not in g_score or gt < g_score[nb]:
                came_from[nb] = cur
                g_score[nb] = gt
                heapq.heappush(open_set, (gt + alpha * h(nb) + beta*cost_to_target(nb, target), gt, nb))
                pushed += 1
                if open_recorder is not None:
                    open_recorder.add(nb)

    storage_cost = pushed
    return None, operation_cost, storage_cost


def astar_with_waypoints(waypoints,env):
    full_path = []
    operation_cost = 0
    peak_storage = 0
    seg_details = []

    with open(PATH, "a", encoding="utf-8") as f:
        for i in range(1, len(waypoints)):
            sub_start = tuple(waypoints[i-1])
            sub_goal = tuple(waypoints[i])

            path, op_cost, st_cost = astar(sub_start, sub_goal, env)

            if path is None:
                print(f"  [FAIL], trying direct line...", file=f)
                if not _is_collision(sub_goal[0], sub_goal[1],
                                     sub_start[0], sub_start[1], env):
                    path = [list(sub_start), list(sub_goal)]
                    op_cost = 0
                    st_cost = 2
                    print(f"         direct line OK",file=f)
                else:
                    print(f"         direct line blocked, skip",file=f)
                    seg_details.append((i, list(sub_start), list(sub_goal), "SKIP", 0, 0))
                    continue

            operation_cost += op_cost-1
            if st_cost > peak_storage:
                peak_storage = st_cost
            seg_details.append((i, list(sub_start), list(sub_goal), "OK", op_cost, st_cost))

            if full_path and path:
                full_path.extend(path[1:])
            else:
                full_path = path

    operation_cost+=1 # 最终目标点需要最终的f-cost计算

    return full_path, operation_cost , peak_storage

# ─── Main ──────────────────────────────────────────────────────────

def run(env,full_waypoints):
    start=env["Start Point"]
    goal=env["Goal Point"]

    if (isinstance(full_waypoints, (list, tuple)) and len(full_waypoints) == 2
            and all(isinstance(value, (int, float)) for value in full_waypoints)):
        candidate_waypoints = [full_waypoints]
    elif isinstance(full_waypoints, (list, tuple)):
        candidate_waypoints = full_waypoints
    else:
        candidate_waypoints = []

    valid_waypoints = []
    for waypoint in candidate_waypoints:
        if (isinstance(waypoint, (list, tuple)) and len(waypoint) == 2
                and all(isinstance(value, (int, float)) for value in waypoint)):
            point = [int(waypoint[0]), int(waypoint[1])]
            if point not in valid_waypoints:
                valid_waypoints.append(point)
        else:
            print(f"Warning: ignoring malformed waypoint: {waypoint!r}")

    full_waypoints = [list(start)] + [
        wp for wp in valid_waypoints
        if wp != start and wp != goal and not on_barrier(wp[0], wp[1], env)
    ] + [list(goal)]

    with open(PATH, "a", encoding="utf-8") as f:
        print(file=f)
        print(f"Waypoints: {full_waypoints}",file=f)

        # ── Baseline: A* direct ──
        print(f"A* Direct:",file=f)
        direct_path, d_op, d_storage = astar(start, goal, env)


        if direct_path:
            d_len = sum(math.sqrt((direct_path[i][0]-direct_path[i-1][0])**2 +
                                  (direct_path[i][1]-direct_path[i-1][1])**2)
                        for i in range(1, len(direct_path)))

            print(f"    point-count={len(direct_path)} pts\t\tlength={d_len:.2f}", file=f)
            print(f"    operation cost   = {d_op}", file=f)
            print(f"    storage cost     = {d_storage}", file=f)
            print(f"    Path: {direct_path}",file=f)
        else:
            print(f"    Result: NO PATH (op={d_op}, storage={d_storage})",file=f)

        # ── A* with waypoints ──
        print(f"LDNN-A*:",file=f)
        wp_path, wp_op, wp_storage = astar_with_waypoints(full_waypoints, env)

        if wp_path:
            w_len = sum(math.sqrt((wp_path[i][0]-wp_path[i-1][0])**2 +
                                  (wp_path[i][1]-wp_path[i-1][1])**2)
                        for i in range(1, len(wp_path)))


            print(f"    point-count={len(wp_path)} pts\t\tlength={w_len:.2f}", file=f)
            print(f"    operation cost        = {wp_op}", file=f)
            print(f"    storage cost          = {wp_storage}", file=f)
            print(f"    Path: {wp_path}", file=f)
        else:
            print(f"\n    Result: FAILED",file=f)

        print(file=f)
        if wp_path:
            if d_storage > 0:
                ratio_s = wp_storage / d_storage
                ratio_o = wp_op / max(1, d_op)
                print(f"  vs Baseline:     storage {ratio_s*100:.1f}%   operation {ratio_o*100:.1f}%", file=f)
        print(file=f)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_TXT = os.path.join(_BASE_DIR, "result", "hybrid2.txt")
ENV_JSON = os.path.join(_BASE_DIR, "problem", "2_标准格式.json")
PLOT_DIR = os.path.join(_BASE_DIR, "result", "plots")


def _parse_hybrid(txt_path):
    comparisons, current = [], None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Waypoints:"):
                if current is not None:
                    comparisons.append(current)
                current = {"waypoints": ast.literal_eval(s[len("Waypoints:"):].strip()), "paths": []}
            elif s.startswith("Path:") and current is not None:
                current["paths"].append(ast.literal_eval(s[len("Path:"):].strip()))
        if current is not None:
            comparisons.append(current)
        return [{"waypoints": c["waypoints"],
               "direct_path": c["paths"][0] if len(c["paths"]) > 0 else None,
               "wp_path": c["paths"][1] if len(c["paths"]) > 1 else None} for c in comparisons]


def _load_envs(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_env(envs, start, goal):
    for e in envs:
        if tuple(e["Start Point"]) == start and tuple(e["Goal Point"]) == goal:
            return e
    return None



def _draw_barriers(ax, env):
    for y, x1, x2 in env.get("Horizontal Barriers", []):
        ax.plot([x1, x2], [y, y], color="black", linewidth=3, solid_capstyle="round")
    for x, y1, y2 in env.get("Vertical Barriers", []):
        ax.plot([x, x], [y1, y2], color="black", linewidth=3, solid_capstyle="round")


def _draw_scene(ax, env, path, open_nodes, waypoints, title, show_waypoints):
    start, goal = waypoints[0], waypoints[-1]
    _draw_barriers(ax, env)
    if open_nodes:
        ax.scatter([n[0] for n in open_nodes], [n[1] for n in open_nodes],
        s=20, marker="s", color="lightgray", zorder=1, label="OpenList")
    if show_waypoints:
        for wp in waypoints[1:-1]:
            ax.scatter(wp[0], wp[1], s=280, marker="*", color="yellow",
                        edgecolors="black", linewidths=0.6, zorder=4, label="Waypoint")
    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path],
                color="#1f77b4", linewidth=1.8, zorder=3, label="Path")
    ax.scatter(start[0], start[1], s=160, color="green", zorder=5, label="Start")
    ax.scatter(goal[0], goal[1], s=160, color="red", zorder=5, label="Goal")
    ax.annotate("Start", (start[0], start[1]), textcoords="offset points", xytext=(8, 8),
                  color="green", fontsize=10, fontweight="bold")
    ax.annotate("Goal", (goal[0], goal[1]), textcoords="offset points", xytext=(8, 8),
                  color="red", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-1, 51); ax.set_ylim(-1, 31); ax.set_aspect("equal")
    ax.set_xticks(range(0, 51, 5)); ax.set_yticks(range(0, 31, 5))
    ax.grid(True, color="gray", linewidth=0.4, alpha=0.35)
    h, l = ax.get_legend_handles_labels()
    by = dict(zip(l, h))
    ax.legend(by.values(), by.keys(), loc="best", fontsize=8, framealpha=0.9)


def plot_comparisons():
    os.makedirs(PLOT_DIR, exist_ok=True)
    envs = _load_envs(ENV_JSON)
    for idx, comp in enumerate(_parse_hybrid(HYBRID_TXT), start=1):
        start, goal = tuple(comp["waypoints"][0]), tuple(comp["waypoints"][-1])
        env = _find_env(envs, start, goal)
        if env is None:
            print(f"[warn] no env for {start}->{goal}"); continue
        direct_open = set(); astar(start, goal, env, open_recorder=direct_open)
        wp_open = set()
        for i in range(1, len(comp["waypoints"])):
            llm_astar(tuple(comp["waypoints"][i-1]), tuple(comp["waypoints"][i]), env, open_recorder=wp_open)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        _draw_scene(axes[0], env, comp["direct_path"], direct_open, comp["waypoints"],
                      "A* Direct", show_waypoints=False)
        _draw_scene(axes[1], env, comp["wp_path"], wp_open, comp["waypoints"],
                      "A* with Waypoints", show_waypoints=True)
        fig.suptitle(f"Comparison {idx}: Start {list(start)} -> Goal {list(goal)}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(PLOT_DIR, f"comparison_{idx:02d}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    plot_comparisons()
