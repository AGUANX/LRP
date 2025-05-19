import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from energy_calculator import (
    calculate_move_energy,
    calculate_total_energy,
    HORIZONTAL_ENERGY_RATE,
    VERTICAL_ENERGY_RATE
)


# 定义欧氏距离函数
def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# 读取CSV文件作为地图数据
def read_map_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        height_map = df.values.astype(np.float32)
        height_map[height_map == -999] = np.nan
        print(f"Map loaded successfully. Shape: {height_map.shape}")
        return height_map
    except Exception as e:
        print(f"Error reading map file: {e}")
        return None


# 获取有效邻居节点（支持8方向）
def get_valid_neighbors(current, height_map, visited, end, wh, wv, max_height_diff_func):
    x, y = current
    current_h = height_map[current]
    candidates = []

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < height_map.shape[0] and 0 <= ny < height_map.shape[1]:
            if (nx, ny) not in visited:
                neighbor_h = height_map[nx, ny]
                if not np.isnan(neighbor_h):
                    max_allowed = max_height_diff_func((nx, ny), end) if max_height_diff_func else 70
                    if abs(neighbor_h - current_h) <= max_allowed:
                        candidates.append((nx, ny))

    priority = []
    end_dx = end[0] - x
    end_dy = end[1] - y
    for node in candidates:
        dx = node[0] - x
        dy = node[1] - y
        dot = dx * end_dx + dy * end_dy
        priority.append(dot)
    sorted_nodes = [node for _, node in sorted(zip(priority, candidates), reverse=True)]
    return sorted_nodes


# 启发式函数
def heuristic(node, current, end, height_map, wh, wv, md_matrix, energy_cache):
    move_energy = calculate_move_energy(current, node, height_map, wh, wv, energy_cache)
    dist_to_end = md_matrix[node]
    height_diff = height_map[node] - height_map[current]

    beta = np.radians(45)
    dx = end[0] - current[0]
    dy = end[1] - current[1]
    distance = euclidean_distance(current, end) + 1e-5
    direction_dot = (node[0] - current[0]) * dx + (node[1] - current[1]) * dy
    direction_bonus = max(0, (np.cos(beta) * direction_dot) / distance) if distance else 0

    return (1 / (move_energy + 1e-10)) * (1 / (dist_to_end + 1)) * direction_bonus


# 选择下一个节点
def choose_next_node(current, height_map, visited, pheromone, alpha, beta, end, wh, wv, md_matrix, energy_cache, max_height_diff_func):
    neighbors = get_valid_neighbors(current, height_map, visited, end, wh, wv, max_height_diff_func)

    if not neighbors:
        return None

    if random.random() < 0.3:
        return random.choice(neighbors)

    probabilities = []
    total = 0.0
    for node in neighbors:
        tau = pheromone[node[0], node[1]] ** alpha
        eta = heuristic(node, current, end, height_map, wh, wv, md_matrix, energy_cache) ** beta
        probabilities.append(tau * eta)
        total += tau * eta

    if total <= 0:
        return random.choice(neighbors)

    probabilities = [p / total for p in probabilities]
    return random.choices(neighbors, weights=probabilities, k=1)[0]


# 高效回溯机制
class SmartBacktracker:
    def __init__(self, jump_step=3):
        self.jump_step = jump_step

    def backtrack(self, path, visited):
        if len(path) > self.jump_step:
            new_path = path[:-self.jump_step]
            removed = path[-self.jump_step:]
            visited = [x for x in visited if x not in removed]
            return new_path, new_path[-1], visited
        return path, path[-1], visited


# 单只蚂蚁的搜索逻辑
def run_ant(params):
    height_map, start, end, alpha, beta, rho, q, wh, wv, md_matrix, energy_cache, max_height_diff_func, backtrack_limit, pheromone = params
    path = [start]
    visited = [start]
    current = start
    backtrack = SmartBacktracker(jump_step=backtrack_limit)
    iteration_steps = 0
    energy_used = 0.0

    while current != end and iteration_steps < 1000:
        next_node = choose_next_node(
            current,
            height_map,
            visited,
            pheromone,
            alpha,
            beta,
            end,
            wh,
            wv,
            md_matrix,
            energy_cache,
            max_height_diff_func
        )

        if next_node is None:
            path, current, visited = backtrack.backtrack(path, visited)
            if current == path[-1]:
                break
            iteration_steps += 1
            continue

        step_energy = calculate_move_energy(current, next_node, height_map, wh, wv, energy_cache)
        energy_used += step_energy
        path.append(next_node)
        visited.append(next_node)
        current = next_node
        iteration_steps += 1

    return path, energy_used


# 蚁群算法
def ant_colony_optimization(height_map, start, end, n_ants=20, iterations=100, alpha=1.2, beta=3.0, rho=0.8, q=200, wh=HORIZONTAL_ENERGY_RATE, wv=VERTICAL_ENERGY_RATE, max_height_diff_func=None, backtrack_limit=10):
    pheromone = np.ones_like(height_map, dtype=float) * 0.01
    best_path = None
    best_cost = float('inf')
    md_matrix = np.zeros_like(height_map)

    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            md_matrix[i, j] = euclidean_distance((i, j), end)

    energy_cache = {}

    best_costs_per_iteration = []
    best_iteration = -1

    pbar = tqdm(total=iterations, desc="ACO Optimization")

    for iteration in range(iterations):
        all_paths = []
        all_costs = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            params_list = [
                (
                    height_map,
                    start,
                    end,
                    alpha,
                    beta,
                    rho,
                    q,
                    wh,
                    wv,
                    md_matrix,
                    energy_cache,
                    max_height_diff_func,
                    backtrack_limit,
                    pheromone.copy()
                )
                for _ in range(n_ants)
            ]
            try:
                results = executor.map(run_ant, params_list)
                for path, cost in results:
                    all_paths.append(path)
                    all_costs.append(cost)
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                continue

        pheromone *= (1 - rho)
        sorted_indices = np.argsort(all_costs)
        elite_paths = [all_paths[i] for i in sorted_indices[:int(len(all_paths) * 0.2)]]

        for path in elite_paths:
            cost = calculate_total_energy(path, height_map, wh, wv, energy_cache)
            if cost < best_cost and path[0] == start and path[-1] == end:
                best_cost = cost
                best_path = path
                best_iteration = iteration + 1
            for node in path:
                pheromone[node[0], node[1]] += q / (cost + 1)

        pbar.update(1)

    pbar.close()

    return best_path if best_path is not None else [], best_cost, best_iteration


# 定义接口函数
def find_optimal_path(map_file, start, end, n_ants=20, iterations=100, alpha=1.2, beta=3.0, rho=0.8, q=200, wh=HORIZONTAL_ENERGY_RATE, wv=VERTICAL_ENERGY_RATE, backtrack_limit=10):
    height_map = read_map_from_csv(map_file)
    best_path, best_cost, _ = ant_colony_optimization(
        height_map,
        start=start,
        end=end,
        n_ants=n_ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        rho=rho,
        q=q,
        wh=wh,
        wv=wv,
        max_height_diff_func=max_height_diff,
        backtrack_limit=backtrack_limit
    )
    return best_path, best_cost


# 定义最大高度差函数
def max_height_diff(current, end):
    return 70


# 选择有效的起点和终点
def select_valid_points(height_map):
    valid_points = []
    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            if not np.isnan(height_map[i, j]):
                valid_points.append((i, j))
    return valid_points


def calculate_path_and_energy(map_file, start_point, end_point):
    """
    计算从起点到终点的最优路径和能耗。

    参数:
        map_file (str): 地图文件的路径。
        start_point (tuple): 起点坐标。
        end_point (tuple): 终点坐标。

    返回:
        tuple: 最优路径和对应的能耗值。
    """
    # 检查地图文件是否存在
    if not os.path.exists(map_file):
        print(f"Error: File {map_file} not found!")
        return None, None

    # 读取地图数据
    height_map = read_map_from_csv(map_file)

    # 检查起点和终点是否有效
    if np.isnan(height_map[start_point]) or np.isnan(height_map[end_point]):
        print(f"Error: Start point {start_point} or end point {end_point} is invalid!")
        return None, None

    # 调用路径规划函数
    best_path, best_cost = find_optimal_path(
        map_file,
        start_point,
        end_point,
        n_ants=50,
        iterations=200,
        alpha=1.5,
        beta=2.5,
        rho=0.7,
        q=100,
        wh=HORIZONTAL_ENERGY_RATE,
        wv=VERTICAL_ENERGY_RATE,
        backtrack_limit=5
    )

    return best_path, best_cost


if __name__ == "__main__":
    map_file = 'convert_data.csv'  # 地图数据文件路径
    start_point = (200, 200)      # 起点坐标
    end_point = (400, 300)        # 终点坐标

    best_path, best_cost = calculate_path_and_energy(map_file, start_point, end_point)

    if best_path and best_path[0] == start_point and best_path[-1] == end_point:
        print(f"Best Path: {best_path}")
        print(f"Best Cost: {best_cost}")
    else:
        print("No valid path found from start to end.")