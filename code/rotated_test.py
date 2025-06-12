'''
这部分是为了计算任务区域的旋转牛耕法
现在差一个问题没有解决，就是寻找返航点时，返航点不再范围内

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from osgeo import gdal
import math
import time

from path_planner import calculate_path_and_energy
from energy_calculator import calculate_total_energy, calculate_move_energy
from tools import matrix_divide, distance_calculate

BATTERY_CAPACITY = 539640  # 无人机的电池容量 (焦耳)

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

    # 变回原来的位置
    p_restored = np.dot(point, R)

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



def calculate_step(path_best, best_angle, hight, cx, cy):
    points = []
    energy = 0
    for i in range(len(path_best)-1):
        new_energy = calculate_move_energy(path_best[i], path_best[i + 1], hight)
        energy += new_energy
        if energy > BATTERY_CAPACITY * 0.8:
            energy = new_energy
            # point = recover(path_best[i], best_angle)
            point = (path_best[i][0] , path_best[i][1])
            # point = recover(point, - best_angle)
            point = (round(point[0]), round(point[1]))
            points.append(point)

    for i in range(len(points)):
        if np.isnan(hight[points[i]]) or hight[points[i]] < 0:
            print("返航点是空")


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



def plot_area(matrix_data, regular_coords, special_coord):
    # 画出区域的范围和坐标戴拿
    plt.figure(figsize=(10, 8))

    # 绘制矩阵区域
    plt.imshow(matrix_data, cmap='gray_r', origin='lower', alpha=0.7)

    # 绘制普通坐标点
    regular_y, regular_x = zip(*regular_coords)
    plt.scatter(regular_x, regular_y, c='blue', s=10, alpha=0.7, label='Back Points')

    # 绘制特殊坐标点
    special_x, special_y = special_coord
    plt.scatter(special_x, special_y, c='red', s=50, edgecolors='black', linewidth=1, label='Nest Point')

    # 添加图例和标签
    plt.legend()
    plt.title('Matrix')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)

    plt.show()

def nest_trans(point, angle):
    angle_rad = math.radians(angle)
    R = np.array([
        [math.cos(angle_rad), math.sin(angle_rad)],
        [-math.sin(angle_rad), math.cos(angle_rad)]
    ])
    R_transpose = R.T
    rotated_point = np.dot(R_transpose, point)
    return rotated_point


def rotated_calculate(matrix, nest_point):
    start_time = time.time()

    Z = matrix
    nrows, ncols = Z.shape
    print(nrows, ncols)
    cx = ncols / 2.0
    cy = nrows / 2.0
    print("cx", cx, "cy", cy)
    # 求一个旋转最大值
    r = math.ceil(math.sqrt((ncols / 2) ** 2 + (nrows / 2) ** 2))
    print('最大距离r：', r)
    points = create_data(r)

    best_length = float('inf')
    best_angle = 0
    path_best = []
    hight_best = []

    # test 最大能耗
    max_length = 0
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
            mask_test = mask
        if length > max_length:
            max_length = length
    print(f"最佳角度: {best_angle}°, 最低能耗: {best_length:.2f}")
    print(f"最高能耗：{max_length:.2f}")
    print(f"Time: {time.time() - start_time:.2f}s")


    # 计算返航点
    no_work_energy = 0
    points_step = calculate_step(path_best, best_angle, hight_best, cx, cy)

    points = []
    for index, row in enumerate(hight_best):
        for j, x in enumerate(row):
            if x > 0 and not pd.isna(x):
                points.append((index, j))
    points = np.array(points)

    # 机巢点处理
    nest = nest_point
    nest = (nest[0] - cx, nest[1] - cy)
    nest = nest_trans(nest, best_angle)
    nest = (nest[0] + r, nest[1] + r)
    nest = (round(nest[0]), round(nest[1]))

    if hight_best[nest] < 0 or np.isnan(hight_best[nest]):
        p = np.array(nest)
        print("机巢点处理", nest)
        distance = np.linalg.norm(points - p, axis=1)
        nest = points[distance.argmin()].tolist()



    for point in points_step:
        if point[0] >= hight_best.shape[0] or point[1] >= hight_best.shape[1]:
            point_list = list(point)

            # 修改值
            if point_list[0] >= hight_best.shape[0] :
                point_list[0] = hight_best.shape[0] - 1
            if point_list[1] >= hight_best.shape[1] :
                point_list[1] = hight_best.shape[1] - 1

            # 转换回元组
            point = tuple(point_list)
        if np.isnan(hight_best[point]) or hight_best[point] < 0:
            p = np.array(point)
            print("返航点处理", point)
            distance = np.linalg.norm(points - p, axis=1)
            point = points[distance.argmin()].tolist()
        energy = calculate_move_energy(tuple(point), tuple(nest), hight_best)
        print(point, "非工作能耗：", energy)
        no_work_energy += energy * 2
    print("返航点", points_step)
    print("机巢点", nest_point, "旋转后", nest)
    plot_area(hight_best, points_step, nest)
    return best_length, max_length, no_work_energy, points_step



# nest_points = ([151, 87], [82, 41], [19, 60], [39, 149], [92, 123])
#
# area_id = pd.read_csv('area_id.csv')
# area_id = area_id.values
# dem = matrix_divide(area_id, 2, nest_points[2])
# rotated_calculate(dem)

