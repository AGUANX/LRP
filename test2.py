import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # 导入 tqdm


# 定义欧氏距离函数
def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# 读取CSV文件作为地图数据
def read_map_from_csv(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path, header=None)  # 假设没有表头
    height_map = df.values.astype(np.float32)  # 转换为 NumPy 数组，并确保是 float32 类型
    # 将 -999 替换为 np.nan，表示无效数据（边界外）
    height_map[height_map == -999] = np.nan
    return height_map


# 计算移动能耗
def calculate_move_energy(p1, p2, height_map, wh, wv, energy_cache):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    key = (p1, p2)

    if key not in energy_cache:
        # 水平移动距离（单位：米）
        horizontal_dist = 30 * np.sqrt(dx ** 2 + dy ** 2)

        # 垂直高度差（单位：米）
        vertical_diff = abs(height_map[p2] - height_map[p1])

        # 计算总能耗
        energy_cache[key] = wh * horizontal_dist + wv * vertical_diff

    return energy_cache[key]


# 计算路径总能耗
def calculate_total_energy(path, height_map, wh, wv, energy_cache):
    if len(path) < 2:
        return 0.0
    return sum(
        calculate_move_energy(path[i], path[i + 1], height_map, wh, wv, energy_cache)
        for i in range(len(path) - 1)
    )


# 获取有效邻居节点（支持8方向）
def get_valid_neighbors(current, height_map, visited, end, wh, wv, max_height_diff_func):
    x, y = current
    current_h = height_map[current]
    candidates = []

    # 8方向
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < height_map.shape[0] and 0 <= ny < height_map.shape[1]:
            if (nx, ny) not in visited:
                neighbor_h = height_map[nx, ny]
                if not np.isnan(neighbor_h):  # 检查是否为边界外
                    max_allowed = max_height_diff_func((nx, ny), end) if max_height_diff_func else 70
                    if abs(neighbor_h - current_h) <= max_allowed:
                        candidates.append((nx, ny))

    # 按终点方向优先级排序
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

    # 方向奖励
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

    if random.random() < 0.3:  # 增加随机探索概率
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
    iteration_steps = 0  # 记录蚂蚁的探索步骤
    energy_used = 0.0  # 记录累积能耗

    while current != end and iteration_steps < 1000:  # 设置最大探索步数
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
def ant_colony_optimization(
        height_map,
        start,
        end,
        n_ants=20,
        iterations=100,
        alpha=1.2,
        beta=3.0,
        rho=0.8,
        q=200,
        wh=1.0,
        wv=5.0,
        max_height_diff_func=None,
        verbose=False,
        backtrack_limit=10
):
    pheromone = np.ones_like(height_map, dtype=float) * 0.01  # 初始化信息素矩阵
    best_path = None
    best_cost = float('inf')
    md_matrix = np.zeros_like(height_map)

    # 初始化曼哈顿距离矩阵
    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            md_matrix[i, j] = euclidean_distance((i, j), end)

    energy_cache = {}

    best_costs_per_iteration = []  # 记录每一代的最佳能耗
    best_iteration = -1  # 记录最优路径所在的代数

    # 创建进度条
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
                    pheromone.copy()  # 每只蚂蚁有独立的信息素副本
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

        # 更新信息素
        pheromone *= (1 - rho)
        sorted_indices = np.argsort(all_costs)
        elite_paths = [all_paths[i] for i in sorted_indices[:int(len(all_paths) * 0.2)]]

        for path in elite_paths:
            cost = calculate_total_energy(path, height_map, wh, wv, energy_cache)
            if cost < best_cost and path[0] == start and path[-1] == end:
                best_cost = cost
                best_path = path
                best_iteration = iteration + 1  # 更新最优路径所在的代数
            for node in path:
                pheromone[node[0], node[1]] += q / (cost + 1)

        # 更新进度条
        pbar.update(1)

    # 关闭进度条
    pbar.close()

    # 确保返回的路径不为 None
    return best_path if best_path is not None else [], best_cost, best_iteration


# 定义最大高度差函数
def max_height_diff(current, end):
    distance = euclidean_distance(current, end)
    return 70 * (1 + np.log1p(distance / 100))