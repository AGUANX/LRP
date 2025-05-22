import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors


points = ((500, 288), (271, 136), (63, 199), (129, 493), (304, 407))

# 读取 CSV 文件
df = pd.read_csv("area_id.csv", header=None)

# 将 DataFrame 转换为 NumPy 数组
data = df.values

# 将空值（NaN）替换为一个特定的值（例如 -1），表示未分配的区域
data = np.nan_to_num(data, nan=-1)

# 获取唯一的 ID 值（包括未分配的区域）
unique_ids = np.unique(data)
print(unique_ids)

# 创建一个自定义的颜色映射
num_ids = len(unique_ids)
colors = []

# 使用 hsv_to_rgb 将 HSV 颜色转换为 RGB 颜色
for i in range(num_ids):
    if i == 0:
        # 第一个颜色为白色，表示未分配的区域
        colors.append([1.0, 1.0, 1.0])  # 白色
    else:
        hsv = [i / num_ids, 1.0, 1.0]  # 色调, 饱和度, 亮度
        rgb = mcolors.hsv_to_rgb(hsv)
        colors.append(rgb)

cmap = ListedColormap(colors)

# 绘制图像
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap=cmap, aspect='auto', vmin=np.min(unique_ids), vmax=np.max(unique_ids))

# 添加颜色条（可选）
plt.colorbar(ticks=unique_ids)
# 绘制点
for point in points:
    plt.scatter(point[1], point[0], color='black', zorder=5)

# 设置标题和标签
plt.title("区域分配图")
plt.xlabel("列")
plt.ylabel("行")

# 显示网格线（可选）
plt.grid(True)

# 显示图像
plt.show()

# 如果需要保存图像
# plt.savefig("area_id_map.png")