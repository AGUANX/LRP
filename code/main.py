import numpy as np
import pandas as pd
from tools import matrix_divide, get_points_list, conversion
from rotated_test import rotated_calculate


df = pd.read_csv("area_id.csv")
data = df.values

# 将空值（NaN）替换为一个特定的值（例如 -1），表示未分配的区域
data = np.nan_to_num(data, nan=-1)

# 获取唯一的 ID 值（包括未分配的区域）
unique_ids = np.unique(data)
print(unique_ids)

# 创建一个自定义的颜色映射
num_ids = len(unique_ids)
colors = []

a = []

nest_points = ([151, 87], [82, 41], [19, 60], [39, 149], [92, 123])
x, shape_low = get_points_list("convert_data_nest.csv")
points_list, shape = get_points_list("convert_data.csv")
nest_points = conversion(shape_low, shape, nest_points)

no_work_energy_list = []

for i in range(num_ids-1):
    print(i)
    dem, nest_point = matrix_divide(data, i, nest_points[i])
    length, k, no_work_energy = rotated_calculate(dem, nest_point)
    a.append(k)
    no_work_energy_list.append(no_work_energy)

for i in range(len(a)):
    print("区域", i ,":", a[i])
    print("区域", i ,"非工作能耗:", no_work_energy_list[i])