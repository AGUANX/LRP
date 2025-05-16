import numpy as np
import math

def generate_mission_coverage_path(height_map, mission_area, angle, max_points=500):
    print(f"Generating coverage path with angle {angle}°")

    x1, y1, x2, y2 = mission_area
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
        fallback_points = []
        for x in range(x1, x2, max(1, (x2 - x1) // 10)):
            for y in range(y1, y2, max(1, (y2 - y1) // 10)):
                if 0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]:
                    if not np.isnan(height_map[x, y]):
                        fallback_points.append((x, y))
        print(f"回退方案生成了 {len(fallback_points)} 个点")
        points = fallback_points

    return points[:min(max_points, len(points))]