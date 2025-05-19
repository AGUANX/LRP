'''
输入convert_data_nest.csv文件，这个文件精度低，计算量少
输出地图的宽高和机巢的选点坐标 地图的宽高是为了对机巢坐标做转换，配合其他精度的地图使用

计划：
将计算过程中离散点和机巢候选点的距离矩阵distance同时输出，用于做任务划分，具体怎么做再distance矩阵处有写
5/19 计划有变，因为机巢选址的地图精度和无人机路径规划的精度不同，筛选距离后续还要精度转换，可以直接复用距离计算函数
'''
import time
import numpy as np
import pandas as pd
import pulp
import random

from tools import get_points_list, distance_calculate
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle



def select(grid_size, num_nest_candidates, seed):
    np.random.seed(seed)

    x_coords = np.random.randint(0, grid_size + 1, size=num_nest_candidates)  # 包含 x_max
    y_coords = np.random.randint(0, grid_size + 1, size=num_nest_candidates)  # 包含 y_max

    # 将 x 和 y 坐标组合成二维数组
    coordinates = np.column_stack((x_coords, y_coords))

    return coordinates





# distance 距离矩阵
# num_nest_candidates  候选点个数
# R 覆盖半径
# num_points 覆盖点个数
# coordinates  候选点
def model(distance, num_nest_candidates, R, num_points, coordinates):
    # 计算覆盖矩阵 A
    A = distance < R
    A = A.astype(int)  # 距离小于R的位置标记为1

    print("A打标记")

    # 创建PuLP问题
    prob = pulp.LpProblem("MinCover", pulp.LpMinimize)

    # 创建变量 z_k（机巢候选点是否被选中）
    z = [pulp.LpVariable(f'z_{k}', cat=pulp.LpBinary) for k in range(num_nest_candidates)]

    # 创建松弛变量（每个离散点一个松弛变量）
    slack_vars = [pulp.LpVariable(f'slack_{m}', lowBound=0, cat=pulp.LpContinuous) for m in range(num_points)]

    # 目标函数：最小化被选中的机巢候选点数量 + 未覆盖点的惩罚
    penalty = 10000  # 未覆盖点的惩罚权重，可以根据实际情况调整
    prob += pulp.lpSum(z) + penalty * pulp.lpSum(slack_vars)

    print("添加约束")
    print(num_points)
    # 添加约束条件：每个离散点必须至少被一个机巢候选点覆盖，或者通过松弛变量允许未覆盖
    for m in range(num_points):
        prob += pulp.lpSum(A[m][k] * z[k] for k in range(num_nest_candidates)) + slack_vars[m] >= 1

    print('求解')
    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # 输出结果
    print(f"目标函数值（被选中的机巢数量）: {int(pulp.value(prob.objective))}")
    selected_nests = [k for k in range(num_nest_candidates) if pulp.value(z[k]) > 0.5]
    print("被选中的机巢候选点索引:")
    print(selected_nests)

    # 验证约束条件是否满足
    coverage = np.array([pulp.value(z[k]) for k in range(num_nest_candidates)])
    A_z = np.dot(A, coverage)
    print("未被完全覆盖的离散点索引（应为空）:")
    print(len(np.where(A_z < 0.999)[0]))  # 0.999用于避免浮点误差

    # 输出离散点和机巢候选点的位置
    pd.set_option('display.max_rows', None)
    nest_candidates_df = pd.DataFrame(coordinates, columns=['X', 'Y'])
    nest_candidates_df['selected'] = [1 if k in selected_nests else 0 for k in range(num_nest_candidates)]
    selected_df = nest_candidates_df[nest_candidates_df['selected'] == 1]

    print("\n机巢候选点坐标及是否被选中:")
    print(selected_df)

    return selected_df, selected_nests



# 画图
def plot_select(boundary, nest_points, coverage_radius):
    """
    画出机巢选点 覆盖范围  以及任务区域范围
    """
    # 创建图形
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制边界
    plt.plot(boundary[:, 0], boundary[:, 1], color='blue', label='边界')
    # 确保边界闭合
    plt.plot([boundary[-1, 0], boundary[0, 0]], [boundary[-1, 1], boundary[0, 1]], color='blue')

    # 绘制机巢点
    for nest_point in nest_points:
        nest_x, nest_y = nest_point
        plt.scatter(nest_x, nest_y, c='red', marker='^', s=100, label='机巢点', zorder=4)

        # 绘制覆盖范围
        coverage_circle = Circle((nest_x, nest_y), radius=coverage_radius,
                                 edgecolor='green', facecolor='none', linestyle='--',
                                 label='覆盖范围' if nest_point == nest_points[0] else "")
        plt.gca().add_patch(coverage_circle)

    # 设置图表样式
    plt.title('坐标列表与机巢点覆盖范围', fontsize=16)
    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('Y坐标', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 显示图形
    plt.show()


# 读取CSV文件
def read_csv(file_path, x_col='x', y_col='y'):
    """
    读取CSV文件并返回坐标数据。

    参数:
    file_path (str): CSV文件的路径。
    x_col (str): x坐标的列名，默认为'x'。
    y_col (str): y坐标的列名，默认为'y'。

    返回:
    X: 包含坐标数据的二维数组。
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查列是否存在
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"CSV文件中不存在列'{x_col}'或'{y_col}'")

        # 检查是否有缺失值
        if df[x_col].isnull().any() or df[y_col].isnull().any():
            print("警告：数据中存在缺失值，已删除包含缺失值的行")
            df = df.dropna(subset=[x_col, y_col])

        # 提取坐标数据
        X = df[[x_col, y_col]].values
        return X

    except FileNotFoundError:
        print(f"错误：文件'{file_path}'不存在")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None


# 对接外部接口
def select_nest(file_path, R, k):
    '''
    输入
    1. 地图信息 是否要进行reshape
    2. R 覆盖半径
    3. k 机巢候选点的个数，可以使用百分比代替
    输出
    1. 机巢选点的坐标
    2. 地图的宽高
    '''
    start_time = time.time()

    # 生成网格点
    points, shape = get_points_list(file_path)
    print("需要覆盖的点数：", len(points))

    # 生成候选点
    random.seed(1234)
    coordinates = random.sample(points, k)
    print(coordinates)
    print("机巢候选点数量：", len(coordinates))

    # 计算距离矩阵
    distance = distance_calculate(points, coordinates)

    print("进入模型")
    #进入模型计算
    select, selected_nests = model(distance, len(coordinates), R, len(points),  coordinates)
    print(select[['X', 'Y']].values)
    print(f"Time: {time.time() - start_time:.2f}s")

    # 筛选距离矩阵
    # ! 新想法，直接复用计算距离函数，这样不用转换精度
    distance = distance[:, selected_nests]
    return select[['X', 'Y']].values, shape, distance, points



'''
nest 是一个机巢点列表
shape 是使用地图的形状
'''
nest, shape, distance, points = select_nest('convert_data_nest.csv', 50, 400)
df = pd.DataFrame(distance)
csv_filename = "distance.csv"
df.to_csv(csv_filename, index=False, header=False)
print(nest, shape, distance)