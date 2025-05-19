'''
重复使用的小工具


'''
import pandas as pd


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

