import numpy as np
from aco import ant_colony_optimization

def find_nearest_charging_station(current_pos, charging_stations):
    if not charging_stations:
        return (0, 0)
    return min(
        charging_stations,
        key=lambda station: euclidean_distance(current_pos, station)
    )

def check_battery_and_recharge(current_pos, charging_stations, height_map, battery_capacity, battery_threshold, aco_params):
    if battery_capacity <= battery_threshold * battery_capacity:
        print(f"电池电量不足: {battery_capacity:.2f}/{battery_capacity} ({battery_capacity / battery_capacity * 100:.1f}%)")
        charging_station = find_nearest_charging_station(current_pos, charging_stations)
        print(f"找到最近的充电站: {charging_station}")
        path_to_station, cost_to_station, _ = ant_colony_optimization(
            height_map,
            current_pos,
            charging_station,
            **aco_params
        )
        if not path_to_station:
            print("警告：无法找到到充电站的路径。尝试另一个站点...")
            if len(charging_stations) > 1:
                charging_station = sorted(
                    charging_stations,
                    key=lambda station: euclidean_distance(current_pos, station)
                )[1]
                path_to_station, cost_to_station, _ = ant_colony_optimization(
                    height_map,
                    current_pos,
                    charging_station,
                    **aco_params
                )
            if not path_to_station:
                print("错误：找不到到任何充电站的路径")
                return [], 0, current_pos, False

        return path_to_station, cost_to_station, charging_station, True

    return [], 0, current_pos, False

def return_to_mission_area(current_pos, resume_pos, height_map, aco_params):
    path_to_resume, cost_to_resume, _ = ant_colony_optimization(
        height_map,
        current_pos,
        resume_pos,
        **aco_params
    )
    if not path_to_resume:
        print(f"警告：无法找到路径返回到任务区域从 {current_pos} 到 {resume_pos}")
        return [], 0
    return path_to_resume[1:], cost_to_resume