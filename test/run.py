from main import EnhancedUAVMissionPlanner
from map_utils import read_map_from_csv
import os
import numpy as np

def main():
    file_path = "convert_data.csv"

    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 未找到!")
        print("使用合成的高度地图进行演示...")
        height_map = np.random.normal(1000, 300, (600, 600))
        from scipy.ndimage import gaussian_filter
        height_map = gaussian_filter(height_map, sigma=5)
    else:
        height_map = read_map_from_csv(file_path)

    start = (0, 0)
    end = (height_map.shape[0] - 1, height_map.shape[1] - 1)

    print(f"地图尺寸: {height_map.shape}")
    print(f"起点: {start}, 终点: {end}")

    planner = EnhancedUAVMissionPlanner(
        height_map=height_map,
        battery_capacity=15000000,
        battery_threshold=0.2,
        num_charging_stations=15,
        num_mission_areas=3,
        simplified=True
    )

    planner.plan_multi_area_mission(start, end)

if __name__ == "__main__":
    main()