这个文件用来记录目录下每个文件的含义

convert_data.csv        高精度地图，用于无人机路径规划

convert_data_nest.csv   低精度地图，降低机巢选址的计算量

distance.csv            记录点到机巢的距离，用于后续任务划分

rotated_test.py         旋转牛耕法

select_nest.py          机巢选址

tools.py                写一些会复用的小工具

计划： 

- [ ] 将距离计算函数写到tools文件并计算运行时间 

- [ ] 任务区域的划分 

- [ ] 旋转牛耕法的适配

- [ ] 模型的入口函数