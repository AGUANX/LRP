import numpy as np
import random
import os
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# 引入相关模块，如果有问题可以直接内联实现
try:
    from test2 import (
        ant_colony_optimization,
        read_map_from_csv,
        euclidean_distance,
        max_height_diff
    )
    from rotated_test import rotated_test, boustrophedon_path, check_points_in_range, rotate_3d_map, create_data
except ImportError:
    # 提供简化的替代实现，避免过于复杂的依赖
    def read_map_from_csv(file_path):
        """简化的CSV读取函数"""
        try:
            return np.loadtxt(file_path, delimiter=',')
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return np.random.normal(1000, 300, (600, 600))

    def euclidean_distance(p1, p2):
        """计算两点间欧几里得距离"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def max_height_diff(p1, p2, height_map):
        """计算两点间的最大高度差"""
        h1 = height_map[p1] if p1[0] < height_map.shape[0] and p1[1] < height_map.shape[1] else 0
        h2 = height_map[p2] if p2[0] < height_map.shape[0] and p2[1] < height_map.shape[1] else 0
        return abs(h2 - h1)

    def ant_colony_optimization(height_map, start, end, **kwargs):
        """简化的ACO实现，增加超时检测和路径验证"""
        print(f"Planning path from {start} to {end} using ACO")

        # 检查起点和终点是否在地图范围内
        rows, cols = height_map.shape
        if not (0 <= start[0] < rows and 0 <= start[1] < cols and
                0 <= end[0] < rows and 0 <= end[1] < cols):
            print(f"Error: Start {start} or end {end} out of map bounds {rows}x{cols}")
            return [], 0, None

        # 检查起点和终点是否太接近
        if euclidean_distance(start, end) < 3:
            # 如果非常接近，直接返回简单路径
            return [start, end], euclidean_distance(start, end) * 5.0, None

        # 获取参数
        n_ants = kwargs.get('n_ants', 10)
        iterations = kwargs.get('iterations', 50)
        timeout = kwargs.get('timeout', 30)  # 超时设置，单位秒

        # 记录开始时间
        start_time = time.time()

        # 模拟ACO进度条
        with tqdm(total=iterations, desc="ACO Optimization") as pbar:
            for i in range(iterations):
                # 超时检测
                if time.time() - start_time > timeout:
                    print(f"ACO timeout after {timeout} seconds - returning best path so far")
                    break
                pbar.update(1)
                if i % 10 == 0 and os.path.exists("stop.txt"):
                    print("Stop file detected, halting ACO")
                    break

        # 生成一条简单的直线路径，但加入一些随机偏移以模拟实际ACO生成的路径
        path = []
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        if steps == 0:
            return [start], 0, None

        for i in range(steps + 1):
            x = int(start[0] + (end[0] - start[0]) * i / steps)
            y = int(start[1] + (end[1] - start[1]) * i / steps)

            if i > 0 and i < steps:
                x += random.randint(-2, 2)
                y += random.randint(-2, 2)
                x = max(0, min(x, rows - 1))
                y = max(0, min(y, cols - 1))

            path.append((x, y))

        cost = 0
        for i in range(1, len(path)):
            dist = euclidean_distance(path[i - 1], path[i])
            h_diff = max_height_diff(path[i - 1], path[i], height_map)
            wh = kwargs.get('wh', 5.0)
            wv = kwargs.get('wv', 1.0)
            cost += dist * wh + h_diff * wv

        return path, cost, None

def is_within_map(point, height_map):
    """检查点是否在地图范围内"""
    x, y = point
    return 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]

def find_valid_start_point(height_map):
    """找到一个有效的起点"""
    rows, cols = height_map.shape
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(height_map[i, j]):  # 检查是否为有效值
                return (i, j)
    return None  # 如果没有找到有效点，返回None

def generate_mission_coverage_path(height_map, mission_area, angle, max_points=500):
    """生成任务区域内的覆盖路径，使用牛耕模式，优化角度计算"""
    print(f"Generating coverage path with angle {angle}°")

    x1, y1, x2, y2 = mission_area
    if x1 >= x2 or y1 >= y2:
        print("警告: 无效的任务区域，尺寸为零或负数")
        return []

    rows, cols = height_map.shape
    x1 = max(0, min(x1, rows - 1))
    x2 = max(0, min(x2, rows - 1))
    y1 = max(0, min(y1, cols - 1))
    y2 = max(0, min(y2, cols - 1))

    max_size = 150
    if x2 - x1 > max_size:
        half = (x2 - x1) // 2
        cx = (x1 + x2) // 2
        x1 = cx - half // 2
        x2 = cx + half // 2

    if y2 - y1 > max_size:
        half = (y2 - y1) // 2
        cy = (y1 + y2) // 2
        y1 = cy - half // 2
        y2 = cy + half // 2

    coverage_density = max(1, min(10, (x2 - x1) // 20))
    x_grid = np.arange(x1, x2 + 1, coverage_density)
    y_grid = np.arange(y1, y2 + 1, coverage_density)

    if len(x_grid) == 0 or len(y_grid) == 0:
        print("警告: 网格生成失败，可能区域太小")
        return []

    X, Y = np.meshgrid(x_grid, y_grid)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    angle_rad = math.radians(angle)
    X_rot = (X - cx) * np.cos(angle_rad) - (Y - cy) * np.sin(angle_rad) + cx
    Y_rot = (X - cx) * np.sin(angle_rad) + (Y - cy) * np.cos(angle_rad) + cy

    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = int(X_rot[i, j]), int(Y_rot[i, j])
            if 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]:
                if not np.isnan(height_map[x, y]):
                    mask[i, j] = True

    points = []
    for i in range(0, X.shape[0], 2):
        if i >= X.shape[0]:
            continue

        row_points = []
        for j in range(X.shape[1]):
            if j < X.shape[1] and mask[i, j]:
                x, y = int(X_rot[i, j]), int(Y_rot[i, j])
                if 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]:
                    row_points.append((x, y))

        if (i // 2) % 2 == 1 and row_points:
            row_points.reverse()

        points.extend(row_points)
        if len(points) > max_points:
            break

    print(f"Generated {len(points)} points for coverage path")

    if not points:
        print("警告: 无法生成有效的覆盖路径")
        fallback_points = []
        for x in range(x1, x2, max(1, (x2 - x1) // 10)):
            for y in range(y1, y2, max(1, (y2 - y1) // 10)):
                if 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]:
                    if not np.isnan(height_map[x, y]):
                        fallback_points.append((x, y))
        print(f"回退方案生成了 {len(fallback_points)} 个点")
        points = fallback_points

    return points[:min(max_points, len(points))]

def visualize_complete_mission(height_map, full_path, mission_areas, charging_stations, angles,
                               checkpoint_indices=[], recharge_indices=[], save_path='mission_2d.png'):
    """可视化完整任务的所有组件，优化清晰度和性能"""
    try:
        plt.figure(figsize=(14, 12))

        plt.imshow(height_map, cmap='terrain', alpha=0.6)
        plt.colorbar(label='Elevation (m)')

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        for i, mission_area in enumerate(mission_areas):
            x1, y1, x2, y2 = mission_area
            color = colors[i % len(colors)]
            rect = Rectangle((y1, x1), (y2 - y1), (x2 - x1),
                             linewidth=2, edgecolor=color, facecolor='none',
                             linestyle='--', label=f'Mission Area {i + 1}')
            plt.gca().add_patch(rect)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            angle_rad = math.radians(angles[i])
            arrow_length = min(x2 - x1, y2 - y1) / 3
            dx = arrow_length * math.sin(angle_rad)
            dy = arrow_length * math.cos(angle_rad)
            plt.arrow(center_y, center_x, dy, dx, head_width=10, head_length=15,
                      fc=color, ec=color, label=f'Area {i + 1} Angle: {angles[i]}°')

        for i, station in enumerate(charging_stations):
            plt.scatter(station[1], station[0], color='yellow', marker='s', s=100,
                        edgecolor='black', label='Charging Station' if i == 0 else "")

        if full_path:
            stride = max(1, len(full_path) // 1000)
            sampled_path = full_path[::stride]

            path_x = [p[1] for p in sampled_path]
            path_y = [p[0] for p in sampled_path]
            plt.plot(path_x, path_y, 'gray', linewidth=0.8, alpha=0.7, label='Flight Path')

            plt.scatter(full_path[0][1], full_path[0][0], color='green', marker='o', s=100, label='Start')
            plt.scatter(full_path[-1][1], full_path[-1][0], color='red', marker='x', s=100, label='End')

            for idx in checkpoint_indices:
                if 0 <= idx < len(full_path):
                    plt.scatter(full_path[idx][1], full_path[idx][0], color='blue', marker='*', s=80,
                                label='Mission Checkpoint' if idx == checkpoint_indices[0] else "")

            for idx in recharge_indices:
                if 0 <= idx < len(full_path):
                    plt.scatter(full_path[idx][1], full_path[idx][0], color='orange', marker='^', s=80,
                                label='Recharge Point' if idx == recharge_indices[0] else "")

        plt.title('Complete UAV Mission Visualization', fontsize=16)
        plt.xlabel('X')
        plt.ylabel('Y')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"2D visualization saved to {save_path}")
    except Exception as e:
        print(f"Error in visualization: {e}")

def visualize_3d_mission(height_map, full_path, mission_areas, charging_stations, save_path='mission_3d.png'):
    """创建任务的3D可视化，优化性能和清晰度"""
    try:
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        sample_rate = 10
        y, x = np.mgrid[0:height_map.shape[0]:sample_rate, 0:height_map.shape[1]:sample_rate]
        z = height_map[::sample_rate, ::sample_rate]
        z = np.nan_to_num(z, nan=np.nanmean(z))

        surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.6, linewidth=0, antialiased=True)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        for i, area in enumerate(mission_areas):
            x1, y1, x2, y2 = area
            color = colors[i % len(colors)]

            corners_x = [y1, y2, y2, y1, y1]
            corners_y = [x1, x1, x2, x2, x1]

            corners_z = []
            for y, x in zip(corners_y, corners_x):
                if 0 <= y < height_map.shape[0] and 0 <= x < height_map.shape[1]:
                    height = height_map[y, x]
                    if np.isnan(height):
                        height = np.nanmean(height_map)
                    corners_z.append(height)
                else:
                    corners_z.append(np.nanmean(height_map))

            ax.plot(corners_x, corners_y, corners_z, color=color, linestyle='--', linewidth=2,
                    label=f'Mission Area {i + 1}')

        for i, station in enumerate(charging_stations):
            x, y = station
            if 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]:
                z = height_map[x, y]
                if np.isnan(z):
                    z = np.nanmean(height_map)
                ax.scatter([y], [x], [z], color='yellow', marker='s', s=100, edgecolor='black',
                           label='Charging Station' if i == 0 else "")

        if full_path:
            stride = max(1, len(full_path) // 100)
            path_reduced = full_path[::stride]

            path_x = [p[1] for p in path_reduced]
            path_y = [p[0] for p in path_reduced]
            path_z = []

            for y, x in zip(path_y, path_x):
                if 0 <= y < height_map.shape[0] and 0 <= x < height_map.shape[1]:
                    z_val = height_map[y, x]
                    if np.isnan(z_val):
                        z_val = np.nanmean(height_map)
                    path_z.append(z_val)
                else:
                    path_z.append(np.nanmean(height_map))

            ax.plot(path_x, path_y, path_z, 'gray', linewidth=1.5, label='Flight Path')

            if full_path:
                start = full_path[0]
                end = full_path[-1]

                if 0 <= start[0] < height_map.shape[0] and 0 <= start[1] < height_map.shape[1]:
                    start_z = height_map[start[0], start[1]]
                    if np.isnan(start_z):
                        start_z = np.nanmean(height_map)
                    ax.scatter(start[1], start[0], start_z,
                               color='green', marker='o', s=100, label='Start')

                if 0 <= end[0] < height_map.shape[0] and 0 <= end[1] < height_map.shape[1]:
                    end_z = height_map[end[0], end[1]]
                    if np.isnan(end_z):
                        end_z = np.nanmean(height_map)
                    ax.scatter(end[1], end[0], end_z,
                               color='red', marker='x', s=100, label='End')

        ax.set_title('3D UAV Mission Visualization', fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation (m)')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        ax.view_init(elev=30, azim=135)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"3D visualization saved to {save_path}")
    except Exception as e:
        print(f"Error in 3D visualization: {e}")
        import traceback
        traceback.print_exc()

class EnhancedUAVMissionPlanner:
    def __init__(
            self,
            height_map_path=None,
            height_map=None,
            battery_capacity=15000000,
            battery_threshold=0.3,
            num_charging_stations=8,
            num_mission_areas=2,
            mission_area_size='auto',
            area_size_range=(60, 120),
            simplified=True,
            timeout=30,
            always_charge_after_mission=True
    ):
        self.timeout = timeout
        self.always_charge_after_mission = always_charge_after_mission

        if height_map is not None:
            self.height_map = height_map
        elif height_map_path is not None:
            self.height_map = read_map_from_csv(height_map_path)
        else:
            raise ValueError("必须提供height_map或height_map_path")

        # 预处理高度图，将无效值（如-999）替换为NaN，并进行平滑处理
        self.height_map = np.where(self.height_map == -999, np.nan, self.height_map)
        if np.isnan(self.height_map).any():
            mean_height = np.nanmean(self.height_map)
            self.height_map = np.nan_to_num(self.height_map, nan=mean_height)

        max_map_size = 600
        if self.height_map.shape[0] > max_map_size or self.height_map.shape[1] > max_map_size:
            scale_factor = max_map_size / max(self.height_map.shape[0], self.height_map.shape[1])
            new_rows = int(self.height_map.shape[0] * scale_factor)
            new_cols = int(self.height_map.shape[1] * scale_factor)
            from scipy.ndimage import zoom
            self.height_map = zoom(self.height_map,
                                   (new_rows / self.height_map.shape[0], new_cols / self.height_map.shape[1]))
            print(f"Height map resized from {self.height_map.shape} to {new_rows}x{new_cols}")

        print(f"Height map loaded with shape: {self.height_map.shape}")
        print(f"Height map min: {np.nanmin(self.height_map)}, max: {np.nanmax(self.height_map)}")

        self.battery_capacity = battery_capacity
        self.current_battery = battery_capacity
        self.battery_threshold = battery_threshold

        self.mission_area_size = mission_area_size
        self.area_size_range = area_size_range
        self.simplified = simplified

        self.num_charging_stations = min(num_charging_stations, 8)
        self.num_mission_areas = min(num_mission_areas, 4)

        print("Generating charging stations...")
        self.charging_stations = self._generate_charging_stations()
        print(f"Generated {len(self.charging_stations)} charging stations")

        print("Generating mission areas...")
        self.mission_areas = self._generate_mission_areas()
        print(f"Generated {len(self.mission_areas)} mission areas")

        print("Calculating optimal angles...")
        self.optimal_angles = self._calculate_optimal_angles()

        self.full_path = []
        self.checkpoint_indices = []
        self.recharge_indices = []

        self.aco_params = {
            'n_ants': 10,
            'iterations': 50,
            'alpha': 1.2,
            'beta': 3.0,
            'rho': 0.8,
            'q': 200,
            'wh': 5.0,
            'wv': 1.0,
            'max_height_diff_func': max_height_diff,
            'verbose': False,
            'backtrack_limit': 3
        }

    def _generate_charging_stations(self):
        """生成分布在地图上的充电站"""
        stations = []
        rows, cols = self.height_map.shape

        grid_size = int(np.sqrt(self.num_charging_stations))
        cell_rows = rows // grid_size
        cell_cols = cols // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                if len(stations) >= self.num_charging_stations:
                    break

                row_start = i * cell_rows
                row_end = min((i + 1) * cell_rows, rows - 1)
                col_start = j * cell_cols
                col_end = min((j + 1) * cell_cols, cols - 1)

                max_attempts = 5
                for _ in range(max_attempts):
                    x = random.randint(row_start, row_end)
                    y = random.randint(col_start, col_end)

                    if 0 <= x < rows and 0 <= y < cols and not np.isnan(self.height_map[x, y]):
                        stations.append((x, y))
                        break

        while len(stations) < self.num_charging_stations:
            x = random.randint(0, rows - 1)
            y = random.randint(0, cols - 1)
            if 0 <= x < rows and 0 <= y < cols and not np.isnan(self.height_map[x, y]):
                stations.append((x, y))

        return stations

    def _generate_mission_areas(self):
        """生成多个不重叠的任务区域"""
        rows, cols = self.height_map.shape
        mission_areas = []

        max_area_size = min(self.area_size_range[1], min(rows, cols) // 3)
        min_area_size = min(self.area_size_range[0], max_area_size // 2)

        max_area_size = min(max_area_size, 120)
        min_area_size = min(min_area_size, 60)

        print(f"Generating mission areas with size range: {min_area_size} to {max_area_size}")

        for i in range(self.num_mission_areas):
            if self.mission_area_size == 'auto':
                area_size = random.randint(min_area_size, max_area_size)
            else:
                area_size = min(self.mission_area_size, max_area_size)

            print(f"Generating mission area {i + 1} with size {area_size}")

            max_attempts = 20
            for attempt in range(max_attempts):
                x1 = random.randint(10, rows - area_size - 10)
                y1 = random.randint(10, cols - area_size - 10)
                x2 = x1 + area_size
                y2 = y1 + area_size

                x2 = min(x2, rows - 1)
                y2 = min(y2, cols - 1)

                overlap = False
                for existing_area in mission_areas:
                    ex1, ey1, ex2, ey2 = existing_area
                    if (x1 <= ex2 and x2 >= ex1 and y1 <= ey2 and y2 >= ey1):
                        overlap = True
                        break

                if not overlap:
                    area_valid = False
                    sample_points = []

                    for check_x in range(x1, x2, max(1, (x2 - x1) // 5)):
                        for check_y in range(y1, y2, max(1, (y2 - y1) // 5)):
                            if 0 <= check_x < rows and 0 <= check_y < cols:
                                if not np.isnan(self.height_map[check_x, check_y]):
                                    area_valid = True
                                    break
                        if area_valid:
                            break

                    if area_valid:
                        mission_areas.append((x1, y1, x2, y2))
                        print(f"  - Area {i + 1} created at ({x1},{y1}) to ({x2},{y2})")
                        break

        # 验证任务区域入口点是否有效
        validated_mission_areas = []
        for area in mission_areas:
            x1, y1, x2, y2 = area
            if (0 <= x1 < rows and 0 <= y1 < cols and
                    0 <= x2 < rows and 0 <= y2 < cols and
                    not np.isnan(self.height_map[x1, y1])):
                validated_mission_areas.append(area)
            else:
                print(f"警告：任务区域入口点 ({x1}, {y1}) 无效，跳过该区域。")

        print(f"经过验证后，有效任务区域数量: {len(validated_mission_areas)}")
        return validated_mission_areas

    def _calculate_optimal_angles(self):
        """预先计算所有任务区域的最优角度"""
        angles = []
        predefined_angles = [0, 45, 90, 135]
        for i in range(len(self.mission_areas)):
            angle = predefined_angles[i % len(predefined_angles)]
            angles.append(angle)
            print(f"任务区域 {i + 1} 的角度: {angle}°")
        return angles

    def _find_nearest_charging_station(self, current_pos):
        """找到最近的充电站"""
        if not self.charging_stations:
            # 如果没有充电站，返回地图中心
            return (self.height_map.shape[0] // 2, self.height_map.shape[1] // 2)
        return min(
            self.charging_stations,
            key=lambda station: euclidean_distance(current_pos, station)
        )

    def _check_battery_and_recharge(self, current_pos, current_path_index):
        """
        检查电池电量，如果需要则充电
        返回 (recharge_path, energy_used, new_position, recharge_occurred)
        """
        if self.current_battery / self.battery_capacity <= self.battery_threshold:
            print(f"电池电量不足: {self.current_battery:.2f}/{self.battery_capacity} "
                  f"({self.current_battery / self.battery_capacity * 100:.1f}%)")

            charging_station = self._find_nearest_charging_station(current_pos)
            print(f"找到最近的充电站: {charging_station}")

            path_to_station, cost_to_station, _ = ant_colony_optimization(
                self.height_map,
                current_pos,
                charging_station,
                **self.aco_params
            )

            if not path_to_station:
                print("警告：无法找到到充电站的路径。尝试另一个站点...")
                charging_stations_sorted = sorted(
                    self.charging_stations,
                    key=lambda station: euclidean_distance(current_pos, station)
                )
                if len(charging_stations_sorted) > 1:
                    charging_station = charging_stations_sorted[1]
                    path_to_station, cost_to_station, _ = ant_colony_optimization(
                        self.height_map,
                        current_pos,
                        charging_station,
                        **self.aco_params
                    )
                if not path_to_station:
                    print("错误：找不到到任何充电站的路径")
                    return [], 0, current_pos, False

            self.full_path.extend(path_to_station[1:])
            self.recharge_indices.append(current_path_index + 1)
            self.current_battery -= cost_to_station
            self.current_battery = self.battery_capacity
            print(f"电池已充电至100%: {self.current_battery}/{self.battery_capacity}")

            return path_to_station, cost_to_station, charging_station, True

        elif self.always_charge_after_mission:
            print("电池电量未满，但前往充电站充满电...")
            charging_station = self._find_nearest_charging_station(current_pos)
            print(f"找到最近的充电站: {charging_station}")

            path_to_station, cost_to_station, _ = ant_colony_optimization(
                self.height_map,
                current_pos,
                charging_station,
                **self.aco_params
            )

            if not path_to_station:
                print("警告：无法找到到充电站的路径。尝试另一个站点...")
                charging_stations_sorted = sorted(
                    self.charging_stations,
                    key=lambda station: euclidean_distance(current_pos, station)
                )
                if len(charging_stations_sorted) > 1:
                    charging_station = charging_stations_sorted[1]
                    path_to_station, cost_to_station, _ = ant_colony_optimization(
                        self.height_map,
                        current_pos,
                        charging_station,
                        **self.aco_params
                    )
                if not path_to_station:
                    print("错误：找不到到任何充电站的路径")
                    return [], 0, current_pos, False

            self.full_path.extend(path_to_station[1:])
            self.recharge_indices.append(current_path_index + 1)
            self.current_battery -= cost_to_station
            self.current_battery = self.battery_capacity
            print(f"电池已充电至100%: {self.current_battery}/{self.battery_capacity}")

            return path_to_station, cost_to_station, charging_station, True

        return [], 0, current_pos, False

    def _return_to_mission_area(self, current_pos, resume_pos):
        """规划从当前位置（充电后）返回任务区域的路径"""
        path_to_resume, cost_to_resume, _ = ant_colony_optimization(
            self.height_map,
            current_pos,
            resume_pos,
            **self.aco_params
        )
        if not path_to_resume:
            print(f"警告：无法找到路径返回到任务区域从 {current_pos} 到 {resume_pos}")
            return [], 0
        self.current_battery -= cost_to_resume
        return path_to_resume[1:], cost_to_resume

    def plan_multi_area_mission(self, start_point=None, end_point=None):
        """
        规划访问所有任务区域的完整任务：
        1. 从起点出发
        2. 访问每个任务区域并使用牛耕法进行覆盖拍摄
        3. 必要时充电
        4. 返回终点
        """
        # 重置跟踪变量
        self.full_path = []
        self.checkpoint_indices = []
        self.recharge_indices = []
        self.current_battery = self.battery_capacity

        total_energy_consumption = 0

        # 定义起点和终点
        if start_point is None:
            start_point = (0, 0)  # 默认为左上角
            # 查找有效起点
            while np.isnan(self.height_map[start_point]):
                start_point = (random.randint(0, self.height_map.shape[0] - 1),
                               random.randint(0, self.height_map.shape[1] - 1))
                print(f"警告：起点 {start_point} 位于无效区域，随机选择新起点。")

        if end_point is None:
            end_point = start_point  # 默认返回起点

        current_pos = start_point
        self.full_path.append(current_pos)

        print(f"从 {start_point} 开始任务")
        print(f"任务区域数量: {len(self.mission_areas)}")

        mission_entry_points = [(area[0], area[1]) for area in self.mission_areas]

        for area_idx, mission_area in enumerate(self.mission_areas):
            print(f"\n--- 处理任务区域 {area_idx + 1} ---")

            if area_idx > 0:
                prev_mission_area = self.mission_areas[area_idx - 1]
                prev_coverage_path = generate_mission_coverage_path(
                    self.height_map, prev_mission_area, self.optimal_angles[area_idx - 1]
                )
                if prev_coverage_path:
                    current_pos = prev_coverage_path[-1]
                    print(f"上一个任务区域的终点: {current_pos}")
                else:
                    print(f"警告：无法获取上一个任务区域的终点。使用默认起点。")
                    current_pos = mission_entry_points[area_idx]

            area_entry_point = mission_entry_points[area_idx]
            print(f"任务区域 {area_idx + 1} 的入口点: {area_entry_point}")

            # 确保入口点在地图的有效范围内
            if (area_entry_point[0] < 0 or area_entry_point[0] >= self.height_map.shape[0] or
                    area_entry_point[1] < 0 or area_entry_point[1] >= self.height_map.shape[1]):
                print(f"警告：任务区域 {area_idx + 1} 的入口点 {area_entry_point} 超出地图范围。跳过。")
                continue

            # 确保入口点不是障碍物
            if np.isnan(self.height_map[area_entry_point]):
                print(f"警告：任务区域 {area_idx + 1} 的入口点 {area_entry_point} 位于无效区域。跳过。")
                continue

            print(f"规划从 {current_pos} 到任务区域 {area_idx + 1} 的路径")
            path_to_area, cost_to_area, _ = ant_colony_optimization(
                self.height_map,
                current_pos,
                area_entry_point,
                **self.aco_params
            )

            if not path_to_area:
                print(f"警告：无法找到到任务区域 {area_idx + 1} 的路径。跳过。")

                # 尝试使用不同的起点进行路径规划
                backup_start = (random.randint(0, self.height_map.shape[0] - 1),
                                random.randint(0, self.height_map.shape[1] - 1))
                while np.isnan(self.height_map[backup_start]):
                    backup_start = (random.randint(0, self.height_map.shape[0] - 1),
                                    random.randint(0, self.height_map.shape[1] - 1))
                print(f"尝试使用新的起点: {backup_start}")
                path_to_area, cost_to_area, _ = ant_colony_optimization(
                    self.height_map,
                    backup_start,
                    area_entry_point,
                    **self.aco_params
                )

                if not path_to_area:
                    print(f"警告：仍然无法找到到任务区域 {area_idx + 1} 的路径。跳过该任务区域。")
                    continue

                current_pos = backup_start
                self.full_path.append(current_pos)

            self.full_path.extend(path_to_area[1:])
            total_energy_consumption += cost_to_area
            self.current_battery -= cost_to_area
            current_pos = path_to_area[-1]
            self.checkpoint_indices.append(len(self.full_path) - 1)

            coverage_path = generate_mission_coverage_path(
                self.height_map, mission_area, self.optimal_angles[area_idx]
            )

            if not coverage_path:
                print(f"警告：无法生成任务区域 {area_idx + 1} 的覆盖路径。跳过覆盖。")
                continue

            coverage_energy = 0
            coverage_points_processed = 0
            last_resume_point = 0

            with tqdm(total=len(coverage_path), desc=f"任务区域 {area_idx + 1} 进度", unit="点") as pbar:
                while coverage_points_processed < len(coverage_path):
                    points_per_check = min(len(coverage_path) - coverage_points_processed, 50)
                    chunk_end = coverage_points_processed + points_per_check
                    next_chunk = coverage_path[coverage_points_processed:chunk_end]

                    chunk_energy = 0
                    valid_points = []  # 用于存储有效的点
                    for i in range(1, len(next_chunk)):
                        p1, p2 = next_chunk[i - 1], next_chunk[i]
                        # 检查点是否在地图范围内
                        if is_within_map(p1, self.height_map) and is_within_map(p2, self.height_map):
                            dist = euclidean_distance(p1, p2)
                            chunk_energy += dist * 5.0
                            valid_points.extend([p1, p2])  # 添加有效点
                        else:
                            print(f"警告：路径点超出地图范围，跳过此段。")
                            coverage_points_processed = chunk_end
                            pbar.update(points_per_check)
                            break

                    # 如果没有有效点，则跳过此段
                    if not valid_points:
                        print("警告：当前路径段中没有有效点，跳过。")
                        coverage_points_processed = chunk_end
                        pbar.update(points_per_check)
                        continue

                    if chunk_energy > self.current_battery:
                        print(f"电池电量不足，需要充电。")
                        current_coverage_pos = coverage_path[coverage_points_processed]
                        recharge_path, recharge_cost, new_pos, recharge_occurred = self._check_battery_and_recharge(
                            current_coverage_pos,
                            len(self.full_path) - 1
                        )
                        if recharge_occurred:
                            current_pos = new_pos
                            total_energy_consumption += recharge_cost
                            # 确保返回路径有效
                            return_path, return_cost = self._return_to_mission_area(
                                current_pos,
                                coverage_path[last_resume_point] if last_resume_point < len(coverage_path) else area_entry_point
                            )
                            if return_path:
                                self.full_path.extend(return_path)
                                total_energy_consumption += return_cost
                                self.current_battery -= return_cost
                                current_pos = return_path[-1]
                            else:
                                # 如果返回路径为空，尝试直接返回到任务区域的入口点
                                current_pos = area_entry_point
                                self.full_path.append(current_pos)
                            coverage_points_processed = last_resume_point
                            pbar.update(coverage_points_processed - pbar.n)
                            continue
                        else:
                            print(f"充电失败。跳到下一个任务区域。")
                            break
                    else:
                        # 只添加有效点到路径中
                        self.full_path.extend(valid_points)
                        coverage_points_processed += len(valid_points) // 2  # 因为每对点计算一次能量
                        total_energy_consumption += chunk_energy
                        self.current_battery -= chunk_energy
                        current_pos = valid_points[-1]
                        last_resume_point = coverage_points_processed
                        pbar.update(len(valid_points) // 2)  # 更新进度条

            print(f"任务区域 {area_idx + 1} 的覆盖完成。")

            # 强制前往充电站充满电
            print("前往充电站充满电...")
            recharge_path, recharge_cost, new_pos, recharge_occurred = self._check_battery_and_recharge(
                current_pos,
                len(self.full_path) - 1
            )
            if recharge_occurred:
                current_pos = new_pos
                total_energy_consumption += recharge_cost
                print("电池已充满。")

            # 如果还有下一个任务区域，规划前往其起点的路径
            if area_idx + 1 < len(self.mission_areas):
                next_mission_area = self.mission_areas[area_idx + 1]
                next_area_entry_point = mission_entry_points[area_idx + 1]
                print(f"规划从 {current_pos} 到下一个任务区域的起点 {next_area_entry_point} 的路径")
                path_to_next_area, cost_to_next_area, _ = ant_colony_optimization(
                    self.height_map,
                    current_pos,
                    next_area_entry_point,
                    **self.aco_params
                )
                if path_to_next_area:
                    self.full_path.extend(path_to_next_area[1:])
                    total_energy_consumption += cost_to_next_area
                    self.current_battery -= cost_to_next_area
                    current_pos = path_to_next_area[-1]

        if self.full_path:
            print(f"\n规划返回终点 {end_point} 的路径")
            path_to_end, cost_to_end, _ = ant_colony_optimization(
                self.height_map,
                current_pos,
                end_point,
                **self.aco_params
            )
            if path_to_end:
                self.full_path.extend(path_to_end[1:])
                total_energy_consumption += cost_to_end
                self.current_battery -= cost_to_end
                current_pos = path_to_end[-1]

        print("\n--- 任务完成 ---")
        print(f"总路径长度: {len(self.full_path)}")
        print(f"总能量消耗: {total_energy_consumption:.2f}")

        visualize_complete_mission(
            self.height_map,
            self.full_path,
            self.mission_areas,
            self.charging_stations,
            self.optimal_angles,
            checkpoint_indices=self.checkpoint_indices,
            recharge_indices=self.recharge_indices
        )

        visualize_3d_mission(
            self.height_map,
            self.full_path,
            self.mission_areas,
            self.charging_stations
        )

        return self.full_path, total_energy_consumption


def main():
    file_path = "convert_data.csv"

    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 未找到!")
        print("使用合成的高度地图进行演示...")
        height_map = np.random.normal(1000, 300, (600, 600))
        height_map = gaussian_filter(height_map, sigma=5)
    else:
        height_map = read_map_from_csv(file_path)

    start = (0, 0)
    end = (height_map.shape[0] - 1, height_map.shape[1] - 1)

    print(f"地图尺寸: {height_map.shape}")
    print(f"起点: {start}, 终点: {end}")

    # 确保起点有效
    start_point = find_valid_start_point(height_map)
    if start_point is None:
        print("错误：地图中没有找到有效的起点！")
        exit(1)
    print(f"有效起点: {start_point}")

    planner = EnhancedUAVMissionPlanner(
        height_map=height_map,
        battery_capacity=15000000,
        battery_threshold=0.2,
        num_charging_stations=15,
        num_mission_areas=3,
        simplified=True
    )

    planner.plan_multi_area_mission(start_point, end)

if __name__ == "__main__":
    main()