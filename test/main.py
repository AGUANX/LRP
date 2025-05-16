import numpy as np
import os
import random
from map_utils import read_map_from_csv, find_valid_start_point
from geometry import euclidean_distance
from aco import ant_colony_optimization
from coverage import generate_mission_coverage_path
from area_manager import generate_mission_areas, calculate_optimal_angles
from energy import find_nearest_charging_station, check_battery_and_recharge, return_to_mission_area
from combined_viz import visualize_mission

class EnhancedUAVMissionPlanner:
    # 初始化无人机和区域设定
    def __init__(self, height_map_path=None, height_map=None, battery_capacity=15000000, battery_threshold=0.3,
                 num_charging_stations=8, num_mission_areas=2, mission_area_size='auto', area_size_range=(60, 120),
                 simplified=True, timeout=30, always_charge_after_mission=True):
        '''
        :param height_map_path:
        :param height_map:  地图高程矩阵
        :param battery_capacity:  无人机电池容量
        :param battery_threshold:   无人机返航电量阈值百分比
        :param num_charging_stations:   ？？？充电桩数量
        :param num_mission_areas:   无人机任务区域个数
        :param mission_area_size:
        :param area_size_range:
        :param simplified:
        :param timeout:
        :param always_charge_after_mission:
        '''
        self.timeout = timeout
        self.always_charge_after_mission = always_charge_after_mission

        if height_map is not None:
            self.height_map = height_map
        elif height_map_path is not None:
            self.height_map = read_map_from_csv(height_map_path)
        else:
            raise ValueError("必须提供height_map或height_map_path")

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
            self.height_map = zoom(self.height_map, (new_rows / self.height_map.shape[0], new_cols / self.height_map.shape[1]))
            print(f"Height map resized from {self.height_map.shape} to {new_rows}x{new_cols}")

        print(f"Height map loaded with shape: {self.height_map.shape}")
        print(f"Height map min: {np.nanmin(self.height_map)}, max: {np.nanmax(self.height_map)}")

        self.battery_capacity = battery_capacity
        self.current_battery = battery_capacity
        self.battery_threshold = battery_threshold

        self.num_charging_stations = min(num_charging_stations, 8)
        self.num_mission_areas = min(num_mission_areas, 4)

        print("Generating charging stations...")
        self.charging_stations = self._generate_charging_stations()
        print(f"Generated {len(self.charging_stations)} charging stations")

        print("Generating mission areas...")
        self.mission_areas = generate_mission_areas(self.height_map, num_mission_areas, area_size_range)
        print(f"Generated {len(self.mission_areas)} mission areas")

        print("Calculating optimal angles...")
        self.optimal_angles = calculate_optimal_angles(len(self.mission_areas))

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
            'wv': 1.0
        }


    def _generate_charging_stations(self):
        '''
        把矩形地图均分为要划分的数量
        然后再均分区域内随机点判断是否可以做充电站

        :return:
        stations : 返回站点列表  充电站
        '''
        stations = []
        rows, cols = self.height_map.shape

        grid_size = int(np.sqrt(self.num_charging_stations))
        cell_rows = rows // grid_size
        cell_cols = cols // grid_size # 向下取整

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


    def plan_multi_area_mission(self, start_point=None, end_point=None):

        self.full_path = []
        self.checkpoint_indices = []
        self.recharge_indices = []
        self.current_battery = self.battery_capacity

        total_energy_consumption = 0

        if start_point is None:
            start_point = find_valid_start_point(self.height_map)
            if start_point is None:
                print("错误：地图中没有找到有效的起点！")
                return

        if end_point is None:
            end_point = start_point

        current_pos = start_point
        self.full_path.append(current_pos)

        print(f"从 {start_point} 开始任务")
        print(f"任务区域数量: {len(self.mission_areas)}")

        mission_entry_points = [(area[0], area[1]) for area in self.mission_areas]

        for area_idx, mission_area in enumerate(self.mission_areas):
            print(f"\n--- 处理任务区域 {area_idx + 1} ---")

            if area_idx > 0:
                prev_mission_area = self.mission_areas[area_idx - 1]
                prev_coverage_path = generate_mission_coverage_path(self.height_map, prev_mission_area, self.optimal_angles[area_idx - 1])
                if prev_coverage_path:
                    current_pos = prev_coverage_path[-1]
                    print(f"上一个任务区域的终点: {current_pos}")
                else:
                    print(f"警告：无法获取上一个任务区域的终点。使用默认起点。")
                    current_pos = mission_entry_points[area_idx]

            area_entry_point = mission_entry_points[area_idx]
            print(f"任务区域 {area_idx + 1} 的入口点: {area_entry_point}")

            print(f"规划从 {current_pos} 到任务区域 {area_idx + 1} 的路径")
            path_to_area, cost_to_area, _ = ant_colony_optimization(
                self.height_map,
                current_pos,
                area_entry_point,
                **self.aco_params
            )

            if not path_to_area:
                print(f"警告：无法找到到任务区域 {area_idx + 1} 的路径。跳过。")
                continue

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

            while coverage_points_processed < len(coverage_path):
                points_per_check = min(len(coverage_path) - coverage_points_processed, 50)
                chunk_end = coverage_points_processed + points_per_check
                next_chunk = coverage_path[coverage_points_processed:chunk_end]

                chunk_energy = 0
                for i in range(1, len(next_chunk)):
                    p1, p2 = next_chunk[i - 1], next_chunk[i]
                    if 0 <= p1[0] < self.height_map.shape[0] and 0 <= p1[1] < self.height_map.shape[1] and \
                       0 <= p2[0] < self.height_map.shape[0] and 0 <= p2[1] < self.height_map.shape[1]:
                        dist = euclidean_distance(p1, p2)
                        chunk_energy += dist * 5.0

                if chunk_energy > self.current_battery:
                    print(f"电池电量不足，需要充电。")
                    current_coverage_pos = coverage_path[coverage_points_processed]
                    recharge_path, recharge_cost, new_pos, recharge_occurred = check_battery_and_recharge(
                        current_coverage_pos,
                        self.charging_stations,
                        self.height_map,
                        self.current_battery,
                        self.battery_threshold,
                        self.aco_params
                    )
                    if recharge_occurred:
                        current_pos = new_pos
                        total_energy_consumption += recharge_cost
                        return_path, return_cost = return_to_mission_area(
                            current_pos,
                            coverage_path[last_resume_point] if last_resume_point < len(coverage_path) else mission_entry_points[area_idx],
                            self.height_map,
                            self.aco_params
                        )
                        if return_path:
                            self.full_path.extend(return_path)
                            total_energy_consumption += return_cost
                            self.current_battery -= return_cost
                            current_pos = return_path[-1]
                        else:
                            current_pos = mission_entry_points[area_idx]
                            self.full_path.append(current_pos)
                        coverage_points_processed = last_resume_point
                    else:
                        print(f"充电失败。跳到下一个任务区域。")
                        break
                else:
                    self.full_path.extend(next_chunk)
                    coverage_points_processed += len(next_chunk) - 1
                    total_energy_consumption += chunk_energy
                    self.current_battery -= chunk_energy
                    current_pos = next_chunk[-1]
                    last_resume_point = coverage_points_processed

            print(f"任务区域 {area_idx + 1} 的覆盖完成。")

            print("前往充电站充满电...")
            recharge_path, recharge_cost, new_pos, recharge_occurred = check_battery_and_recharge(
                current_pos,
                self.charging_stations,
                self.height_map,
                self.current_battery,
                self.battery_threshold,
                self.aco_params
            )
            if recharge_occurred:
                current_pos = new_pos
                total_energy_consumption += recharge_cost
                print("电池已充满。")

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

        # 调用合并后的可视化函数
        visualize_mission(
            self.height_map,
            self.full_path,
            self.mission_areas,
            self.charging_stations,
            self.optimal_angles,
            checkpoint_indices=self.checkpoint_indices,
            recharge_indices=self.recharge_indices
        )

        return self.full_path, total_energy_consumption