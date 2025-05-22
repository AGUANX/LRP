'''
重复使用的小工具


'''
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix



def get_points_list(file_path):
    '''
    获取范围可用坐标点, 返回包括可用坐标的一维列表
    :param file_path: 文件路径
    :return:
    points 一维列表可用坐标点
    df.shape 读取地图的范围
    '''
    df = pd.read_csv(file_path)
    points = []
    for index, row in df.iterrows():
        for j, x in enumerate(row):
            if x != -999 and not pd.isna(x):
                points.append((index, j))

    return points, df.shape

def distance_calculate(points, coordinates):
    '''
    计算两个坐标点列表的距离，输入n和m个坐标，输出n*m个距离
    '''
    distances = distance_matrix(points, coordinates)
    return distances


def conversion(shape1, shape2, points):
    '''
    坐标系转换，输入原坐标系宽高，转换后坐标系宽高，坐标点，输出转换后坐标
    :param shape1:
    :param shape2:
    :param points:
    :return:
    '''
    # 计算缩放比例
    x_scale = shape2[0] / shape1[0]
    y_scale = shape2[1] / shape1[1]
    print(f"x_scale: {x_scale}, y_scale: {y_scale}")

    # 假设points是一个包含多个点的元组列表，每个点有x和y坐标
    # 对每个点进行坐标转换
    converted_points = []
    for point in points:
        x = round(point[0] * x_scale)
        y = round(point[1] * y_scale)
        converted_points.append((x, y))
    return converted_points



def matrix_divide(matrix, i, point):
    # 找到矩阵中值为 1 的行和列的索引
    rows_with_1 = np.where(np.any(matrix == i, axis=1))[0]
    cols_with_1 = np.where(np.any(matrix == i, axis=0))[0]

    # 找到行和列索引的最小和最大值，确定包含所有 1 的子矩阵范围
    min_row, max_row = np.min(rows_with_1), np.max(rows_with_1)
    min_col, max_col = np.min(cols_with_1), np.max(cols_with_1)

    dem = pd.read_csv("convert_data.csv")
    dem_data = dem.values
    dem_data = dem_data[min_row:max_row + 1, min_col:max_col + 1]

    # 提取包含所有 1 的子矩阵
    submatrix_with_1 = matrix[min_row:max_row + 1, min_col:max_col + 1]

    # 将子矩阵中的非 1 值替换为 0
    submatrix_with_1[submatrix_with_1 != i] = None
    dem_data[submatrix_with_1 != i] = None

    nest_point = [point[0] - min_row, point[1] - min_col]
    print("原始矩阵:")
    print(matrix)
    print("\n提取并处理后的子矩阵:")
    return dem_data, nest_point