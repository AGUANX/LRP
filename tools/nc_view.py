from netCDF4 import Dataset
import numpy as np

# 打开.nc 文件
file_path = '../HiTMC_Monthly_China_AVP_200501_200512.nc'  # 替换为你的文件路径
nc_file = Dataset(file_path, 'r')  # 'r' 表示只读模式

# 查看文件信息
print('文件名:', file_path)
print('文件格式:', nc_file.file_format)
print('文件标题:', getattr(nc_file, 'title', 'No title found'))

# 查看文件中的变量
print('\n变量列表:')
for var_name in nc_file.variables:
    var = nc_file.variables[var_name]
    print('变量名:', var_name)
    print('维度:', var.dimensions)
    print('数据类型:', var.dtype)
    print('变量形状:', var.shape)
    print('变量属性:')
    for attr_name in var.ncattrs():
        print(f'  {attr_name}: {getattr(var, attr_name)}')
    print('-----------------------------')

# 查看文件中的维度
print('\n维度列表:')
for dim_name in nc_file.dimensions:
    dim = nc_file.dimensions[dim_name]
    print('维度名:', dim_name)
    print('维度大小:', len(dim))
    print('是否无限维度:', dim.isunlimited())
    print('-----------------------------')

# 读取并处理某个变量数据（以 “temperature” 变量为例）
if 'temperature' in nc_file.variables:
    temp_var = nc_file.variables['temperature']
    temp_data = temp_var[:]  # 获取变量的所有数据
    print('\n读取的 temperature 数据形状:', temp_data.shape)
    print('读取的 temperature 数据类型:', temp_data.dtype)
    print('温度数据前几个值:', temp_data.flatten()[:5])  # 展平数组并查看前几个值
else:
    print('\n文件中没有找到 temperature 变量')

# 关闭文件
nc_file.close()