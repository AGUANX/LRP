这个文件用来记录目录下每个文件的含义

area_id.csv             记录高精度坐标点属于机巢的划分
convert_data.csv        高精度地图，用于无人机路径规划
convert_data_nest.csv   低精度地图，降低机巢选址的计算量
distance.csv            记录点到机巢的距离，用于后续任务划分
divide_area.py          任务划分
rotated_test.py         旋转牛耕法
select_nest.py          机巢选址
tools.py                写一些会复用的小工具

