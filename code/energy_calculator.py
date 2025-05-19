import numpy as np
DRONE_WEIGHT = 1.85  # 无人机总重量 (千克)
BATTERY_CAPACITY = 539640  # 无人机的电池容量 (焦耳)
AIR_DENSITY = 1.205  # 空气密度
FRONTAL_AREA = 0.0884  # 正面面积 (平方米)
DRONE_WIDTH = 0.1462  # 无人机的宽度 (米)
HORIZONTAL_SPEED = 12  # 无人机的水平飞行速度 (米/秒)
AIR_RESISTANCE_COEFFICIENT = 0.47  # 空气阻力系数
VERTICAL_ENERGY_RATE = 0.518  # 无人机的垂直飞行能耗率 (焦耳/米)
HORIZONTAL_ENERGY_RATE = 12.86  # 无人机的水平飞行能耗率 (焦耳/米)


# 无人机参数


def calculate_move_energy(p1, p2, height_map, wh=HORIZONTAL_ENERGY_RATE, wv=VERTICAL_ENERGY_RATE, energy_cache=None):
    """
    计算从点 p1 到点 p2 的移动能耗。

    参数:
        p1 (tuple): 起点坐标
        p2 (tuple): 终点坐标
        height_map (np.ndarray): 高度地图
        wh (float): 水平能耗率
        wv (float): 垂直能耗率
        energy_cache (dict): 能耗缓存

    返回:
        float: 从 p1 到 p2 的能耗
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    key = (p1, p2)

    if energy_cache is None:
        energy_cache = {}

    if key not in energy_cache:
        # 水平移动距离（单位：米）
        horizontal_dist = 30 * np.sqrt(dx ** 2 + dy ** 2)

        # 垂直高度差（单位：米）
        vertical_diff = abs(height_map[p2] - height_map[p1])

        # 计算总能耗
        energy_cache[key] = wh * horizontal_dist + wv * vertical_diff

    return energy_cache[key]


def calculate_total_energy(path, height_map, wh=HORIZONTAL_ENERGY_RATE, wv=VERTICAL_ENERGY_RATE, energy_cache=None):
    """
    计算路径的总能耗。

    参数:
        path (list): 路径点列表
        height_map (np.ndarray): 高度地图
        wh (float): 水平能耗率
        wv (float): 垂直能耗率
        energy_cache (dict): 能耗缓存

    返回:
        float: 路径的总能耗
    """
    if len(path) < 2:
        return 0.0

    if energy_cache is None:
        energy_cache = {}

    return sum(
        calculate_move_energy(path[i], path[i + 1], height_map, wh, wv, energy_cache)
        for i in range(len(path) - 1)
    )


def calculate_flight_time(energy_consumed):
    """
    计算无人机的飞行时间。

    参数:
        energy_consumed (float): 消耗的能耗

    返回:
        float: 飞行时间（秒）
    """
    return energy_consumed / BATTERY_CAPACITY


def calculate_max_flight_distance():
    """
    计算无人机的最大飞行距离。

    返回:
        tuple: 最大水平飞行距离和最大垂直飞行距离（米）
    """
    max_horizontal_distance = BATTERY_CAPACITY / HORIZONTAL_ENERGY_RATE
    max_vertical_distance = BATTERY_CAPACITY / VERTICAL_ENERGY_RATE
    return max_horizontal_distance, max_vertical_distance


def calculate_remaining_energy(initial_energy, energy_consumed):
    """
    计算无人机的剩余电量。

    参数:
        initial_energy (float): 初始电量
        energy_consumed (float): 消耗的电量

    返回:
        float: 剩余电量
    """
    return initial_energy - energy_consumed