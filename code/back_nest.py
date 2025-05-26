import numpy as np


def energy_nest(start, end):
    # 计算距离，
    distance = np.linalg.norm(start - end)
    # 能耗模型
