'''
这部分是为了计算任务区域的旋转牛耕法   还没模块化   要把mian函数改成接口
计划：
1. 接口输入单个任务区域的数据，高程矩阵  不需要覆盖的地方是nan
2. 输出一个角度和电量消耗
3. 把能耗模型改成外接模型，使用另外的文件去做
'''

import numpy as np
import pandas as pd
from osgeo import gdal
import math
import time

from path_planner import calculate_path_and_energy
from energy_calculator import calculate_total_energy, calculate_move_energy
from tools import matrix_divide



BATTERY_CAPACITY = 539640  # 无人机的电池容量 (焦耳)
class UAV:
    # 能耗系数 水平能耗k_s 垂直能耗 k_c
    k_s = 0.0039
    k_c = 0.0029


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
    '''
    旋转点
    :param points:
    :param angle:
    :return:
    '''
    angle_rad = math.radians(angle)
    R = np.array([
        [math.cos(angle_rad), math.sin(angle_rad)],
        [-math.sin(angle_rad), math.cos(angle_rad)]
    ])
    rotated_points = np.dot(points, R)
    size = int(math.sqrt(len(points)))  # 2r+1
    return rotated_points[:, 0].reshape(size, size), rotated_points[:, 1].reshape(size, size)



def recover(point, angle_rad):
    # 定义旋转矩阵
    R = np.array([
        [math.cos(angle_rad), math.sin(angle_rad)],
        [-math.sin(angle_rad), math.cos(angle_rad)]
    ])

    # 计算逆矩阵（转置矩阵）
    R_inv = R.T

    # 变回原来的位置
    p_restored = R_inv @ point
    return p_restored


def get_near_data(x, y, nrows, ncols):
    '''
    获取xy附近四个点的坐标
    '''
    x_floor = int(np.floor(x))
    x_ceil = min(x_floor + 1, ncols - 1)
    y_floor = int(np.floor(y))
    y_ceil = min(y_floor + 1, nrows - 1)
    return x_floor, y_floor, x_ceil, y_ceil



def check_points_in_range(X_rotated, Y_rotated, nrows, ncols):
    '''
    检查旋转点是否再范围内
    '''
    mask = (X_rotated >= 0) & (X_rotated < ncols) & (Y_rotated >= 0) & (Y_rotated < nrows)
    return mask




def hight_interpolation(X_rotated, Y_rotated, nrows, ncols, Z):
    mask = check_points_in_range(X_rotated, Y_rotated, nrows, ncols)
    hight = np.zeros_like(X_rotated, dtype=float)

    for i in range(X_rotated.shape[0]):
        for j in range(X_rotated.shape[1]):
            if mask[i, j]:
                x = X_rotated[i][j]
                y = Y_rotated[i][j]
                x_floor, y_floor, x_ceil, y_ceil = get_near_data(x, y, nrows, ncols)

                dx = x - x_floor
                dy = y - y_floor

                h1 = Z[y_floor, x_floor]
                h2 = Z[y_floor, x_ceil]
                h3 = Z[y_ceil, x_floor]
                h4 = Z[y_ceil, x_ceil]

                if not pd.isna(h1) and not pd.isna(h2) and not pd.isna(h3) and not pd.isna(h4):
                    hight[i][j] = (h1 * (1 - dx) * (1 - dy) +
                                   h2 * dx * (1 - dy) +
                                   h3 * (1 - dx) * dy +
                                   h4 * dx * dy)
                else:
                    hight[i][j] = None
                    mask[i][j] = False
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
    return path


def energyConsumption(dx, dy, dz, k_s, k_c):
    # 能耗系数 水平能耗k_s 垂直能耗 k_c


    # 计算水平移动距离（勾股定理）
    horizontal_distance = math.sqrt(dx ** 2 + dy ** 2) * 17

    # 计算水平能耗
    horizontal_energy = horizontal_distance * k_s

    # 计算垂直能耗（取绝对值后计算）
    vertical_energy = abs(dz) * k_c

    # 总能耗 = 水平 + 垂直
    total_energy = horizontal_energy + vertical_energy

    return round(total_energy, 4)  # 保留四位小数



def calculate_step(path_best, best_angle, hight):
    points = []
    energy = 0
    for i in range(len(path_best)-1):
        new_energy = calculate_move_energy(path_best[i], path_best[i + 1], hight)
        energy += new_energy
        if energy > BATTERY_CAPACITY * 0.8:
            energy = new_energy
            points.append(path_best[i])

    for i in range(len(points)):
        points[i] = recover(points[i], best_angle)
        points[i] = (round(points[i][0]), round(points[i][1]))

    return points



def calculate_path(path, hight, k_s, k_c):
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
        total += energyConsumption(dx, dy, dz, k_s, k_c)
    return total


def rotated_calculate(matrix, nest_point):
    start_time = time.time()

    Z = matrix
    nrows, ncols = Z.shape
    print(nrows, ncols)
    cx = ncols / 2.0
    cy = nrows / 2.0
    # 求一个旋转最大值
    r = math.ceil(math.sqrt((ncols / 2) ** 2 + (nrows / 2) ** 2))
    print('最大距离r：', r)
    points = create_data(r)

    best_length = float('inf')
    best_angle = 0
    path_best = []
    hight_best = []

    u = UAV()
    # test 最大能耗
    k = 0
    for angle in range(0, 180):
        # 获取点旋转之后对应的点
        dx_rot, dy_rot = rotate_3d_map(points, angle)
        X_rot = dx_rot + cx
        Y_rot = dy_rot + cy
        # 检查加插值
        mask, hight = hight_interpolation(X_rot, Y_rot, nrows, ncols, Z)
        # 牛耕
        path = boustrophedon_path(mask)
        # 计算能耗
        length = calculate_total_energy(path, hight)
        print(length)

        if length < best_length:
            best_length = length
            best_angle = angle
            path_best = path
            hight_best = hight
        if length > k:
            k = length
    print(f"最佳角度: {best_angle}°, 最低能耗: {best_length:.2f}")
    print(f"最高能耗：{k:.2f}")
    print(f"Time: {time.time() - start_time:.2f}s")


    # 计算返航点
    no_work_energy = 0
    points_step = calculate_step(path_best, best_angle, hight_best)
    for point in points_step:
        print(point, nest_point, matrix.shape)
        path, energy = calculate_path_and_energy(matrix, point, nest_point)
        no_work_energy += energy * 2
    return best_length, k, no_work_energy



# nest_points = ([151, 87], [82, 41], [19, 60], [39, 149], [92, 123])
#
# area_id = pd.read_csv('area_id.csv')
# area_id = area_id.values
# dem = matrix_divide(area_id, 2, nest_points[2])
# rotated_calculate(dem)

