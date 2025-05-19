import os
import numpy as np  # 导入 numpy 库
from path_planner import find_optimal_path, read_map_from_csv
from energy_calculator import HORIZONTAL_ENERGY_RATE, VERTICAL_ENERGY_RATE


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
        n_ants=20,
        iterations=50,
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
    start_point = (200, 200)  # 起点坐标
    end_point = (300, 300)  # 终点坐标

    best_path, best_cost = calculate_path_and_energy(map_file, start_point, end_point)

    if best_path and best_path[0] == start_point and best_path[-1] == end_point:
        print(f"Best Path: {best_path}")
        print(f"Best Cost: {best_cost}")
    else:
        print("No valid path found from start to end.")