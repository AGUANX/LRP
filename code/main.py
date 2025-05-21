import numpy as np
import pandas as pd
from tools import matrix_divide
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

for i in range(num_ids-1):
    print(i)
    dem = matrix_divide(data, i)
    rotated_calculate(dem)
    p