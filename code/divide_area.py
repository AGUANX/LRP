'''
这部分用于划分任务区域，原理是坐标点距离哪个机巢近，坐标点就归属于哪个机巢

后续可以把任务区域划分画出来
'''
import time
import numpy as np
import pandas as pd

from tools import get_points_list, distance_calculate, conversion


# 一维转二维
def to2(area_id, points, shape):
    matrix = [[None for _ in range(shape[1])] for _ in range(shape[0])]
    for index in range(len(area_id)):
        matrix[points[index][0]][points[index][1]] = area_id[index]

    return matrix



def divide(file_path, nest_points, nest_shape):
    # 获取高精度地图的坐标列表
    points_list, shape = get_points_list(file_path)
    print(shape_low)

    # 转换机巢选点精度
    coordinates = conversion(nest_shape, shape, nest_points)

    # 封装
    points = np.array(points_list)
    coordinates = np.array(coordinates)

    # 计算距离
    start_time = time.time()
    distance = distance_calculate(points, coordinates)
    end_time = time.time()
    print("计算矩阵矩阵时间", end_time - start_time)

    # 给每个坐标点赋予一个id，表示机巢的选择
    area_id = []
    for index in range(len(points)):
        id = np.argmin(distance[index])
        area_id.append(id)

    print(points.shape)

    # 将一维area_di转换成二维
    area = to2(area_id, points, shape)

    return area




x, shape_low = get_points_list("convert_data_nest.csv")
nest_points = ([151, 87], [82, 41], [19, 60], [39, 149], [92, 123])
area_id = divide("convert_data.csv", nest_points, shape_low)
df = pd.DataFrame(area_id)
df.to_csv("area_id.csv", index=False, header=False)
print(area_id)