# rotated_test.py
import numpy as np
from osgeo import gdal
import math
import time


def read_dem_data(file_path):
    gdal.AllRegister()
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    ncols = dataset.RasterXSize
    nrows = dataset.RasterYSize
    print(ncols, nrows)
    Z = band.ReadAsArray(0, 0, ncols, nrows)
    return Z, nrows, ncols


def create_data(r):
    dx = np.arange(-r, r + 1)
    dy = np.arange(-r, r + 1)
    DX, DY = np.meshgrid(dx, dy)
    points = np.array([DX.flatten(), DY.flatten()]).T
    return points


def rotate_3d_map(points, angle):
    angle_rad = math.radians(angle)
    R = np.array([
        [math.cos(angle_rad), math.sin(angle_rad)],
        [-math.sin(angle_rad), math.cos(angle_rad)]
    ])
    rotated_points = np.dot(points, R)
    size = int(math.sqrt(len(points)))  # 2r+1
    return rotated_points[:, 0].reshape(size, size), rotated_points[:, 1].reshape(size, size)


def check_points_in_range(X_rotated, Y_rotated, nrows, ncols, Z):
    mask = (X_rotated >= 0) & (X_rotated < ncols) & (Y_rotated >= 0) & (Y_rotated < nrows)
    hight = np.zeros_like(X_rotated, dtype=float)

    for i in range(X_rotated.shape[0]):
        for j in range(X_rotated.shape[1]):
            if mask[i][j]:
                x = X_rotated[i][j]
                y = Y_rotated[i][j]
                x_floor = int(np.floor(x))
                x_ceil = min(x_floor + 1, ncols - 1)
                y_floor = int(np.floor(y))
                y_ceil = min(y_floor + 1, nrows - 1)

                dx = x - x_floor
                dy = y - y_floor

                h1 = Z[y_floor, x_floor]
                h2 = Z[y_floor, x_ceil]
                h3 = Z[y_ceil, x_floor]
                h4 = Z[y_ceil, x_ceil]

                hight[i][j] = (h1 * (1 - dx) * (1 - dy) +
                                h2 * dx * (1 - dy) +
                                h3 * (1 - dx) * dy +
                                h4 * dx * dy)
            else:
                hight[i][j] = 0.0
    return mask, hight


def boustrophedon_path(mask):
    path = []
    for i in range(mask.shape[0]):
        if i % 2 == 0:
            for j in range(mask.shape[1]):
                if mask[i][j]:
                    path.append((i, j))
        else:
            for j in range(mask.shape[1] - 1, -1, -1):
                if mask[i][j]:
                    path.append((i, j))
    print(len(path))
    return path


def energyConsumption(dx, dy, dz):

    # 能耗系数 水平能耗k_s 垂直能耗 k_c
    k_s = 0.1
    k_c = 0.03

    # 计算水平移动距离（勾股定理）
    horizontal_distance = math.sqrt(dx ** 2 + dy ** 2)

    # 计算水平能耗
    horizontal_energy = horizontal_distance * k_s

    # 计算垂直能耗（取绝对值后计算）
    vertical_energy = abs(dz) * k_c

    # 总能耗 = 水平 + 垂直
    total_energy = horizontal_energy + vertical_energy

    return round(total_energy, 4)  # 保留四位小数


def calculate_path(path, hight):
    if not path:
        return 0
    total = 0.0
    for k in range(1, len(path)):
        i1, j1 = path[k - 1]
        i2, j2 = path[k]
        z1 = hight[i1][j1]
        z2 = hight[i2][j2]
        dx = i2 - i1
        dy = j2 - j1
        dz = z2 - z1
        total += energyConsumption(dx, dy, dz)
    return total


def rotated_test(region_bounds, dem_data):
    # region: 任务区域 T_k
    # 返回最优航向角 theta_k 和覆盖路径能耗 E_cov
    x1, y1, x2, y2 = region_bounds
    Z, nrows, ncols = dem_data
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = max(x2 - x1, y2 - y1) // 2
    points = create_data(r)

    best_length = float('inf')
    best_angle = 0

    # test 最大能耗
    k = 0
    for angle in range(0, 180):
        dx_rot, dy_rot = rotate_3d_map(points, angle)
        X_rot = dx_rot + cx
        Y_rot = dy_rot + cy
        mask, hight = check_points_in_range(X_rot, Y_rot, nrows, ncols, Z)
        path = boustrophedon_path(mask)
        length = calculate_path(path, hight)

        # 检查计算结果是否为有效值
        if np.isnan(length):
            print(f"Warning: NaN value detected at angle {angle}°, skipping.")
            continue

        print(length)

        if length < best_length:
            best_length = length
            best_angle = angle
        if length > k:
            k = length
    print(f"最佳角度: {best_angle}°, 最低能耗: {best_length:.2f}")
    print(f"最高能耗：{k:.2f}")
    return best_angle, best_length