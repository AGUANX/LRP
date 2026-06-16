# 🚁 UAV-LRP-Solver

**无人机选址-路径问题（LRP）求解工具**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

本项目针对无人机（UAV）场景下的**选址-路径问题（Location-Routing Problem, LRP）**，提供了一套完整的求解框架。主要解决以下核心问题：

- **机巢选址**：在给定区域内确定无人机机巢（基站）的最佳位置
- **任务区域划分**：根据机巢位置将任务区域合理分配给各机巢
- **路径规划**：为每架无人机规划覆盖任务区域的飞行路径
- **能耗评估**：计算无人机执行任务的能耗与返航需求

项目包含高精度与低精度两种地图处理模式，支持在降低计算量的同时保证选址精度[reference:2]。

---

## 🏗️ 系统架构

```
LRP/
├── code/                          # 核心代码目录
│   ├── main.py                    # 主程序入口
│   ├── divide_area.py             # 任务区域划分（最近机巢原则）
│   ├── select_nest.py             # 机巢选址算法
│   ├── path_planner.py            # 路径规划（蚁群算法 ACO）
│   ├── energy_calculator.py       # 能耗计算模型
│   ├── rotated_test.py            # 旋转牛耕法（覆盖路径生成）
│   ├── tools.py                   # 通用工具函数
│   ├── back_nest.py               # 返航点计算
│   ├── convert_data.csv           # 高精度地图数据
│   ├── convert_data_nest.csv      # 低精度地图数据（用于选址）
│   ├── area_id.csv                # 区域划分结果
│   └── distance.csv               # 点到机巢距离矩阵
├── test/                          # 测试文件
├── tools/                         # 辅助工具
└── 1.py                           # 早期版本（含ACO实现）
```

---

## ✨ 核心功能

### 1. 机巢选址 (`select_nest.py`)
基于低精度地图快速确定机巢候选位置，降低选址阶段的计算复杂度[reference:3]。

### 2. 任务区域划分 (`divide_area.py`)
采用**最近邻原则**：每个坐标点归属于距离最近的机巢，实现任务区域的合理划分[reference:4]。

### 3. 路径规划 (`path_planner.py`)
基于**蚁群算法（Ant Colony Optimization）** 规划两点间的最优路径，综合考虑：
- 水平移动能耗
- 垂直爬升/下降能耗
- 地形高度约束[reference:5][reference:6]

### 4. 旋转牛耕法覆盖 (`rotated_test.py`)
采用**旋转牛耕法（Rotated Boustrophedon）** 生成区域全覆盖路径，并计算：
- 各区域最大路径长度
- 非工作能耗（返航能耗）
- 返航次数与返航点[reference:7][reference:8]

### 5. 能耗评估 (`energy_calculator.py`)
内置无人机能耗模型，支持：
- 单步移动能耗计算
- 整条路径总能耗评估
- 电池容量约束检查（默认 539640 J）[reference:9]



### 数据准备
- `convert_data.csv`：高精度高程地图
- `convert_data_nest.csv`：低精度地图（用于机巢选址）
- 地图中以 `-999` 表示不可用区域（障碍/边界）[reference:10]

---

## 📊 输出结果

程序运行后会输出：
- 每个任务区域的最大路径长度
- 每个任务区域的非工作能耗（返航能耗）
- 每个任务区域的返航次数[reference:11]

---

## 📁 主要模块说明

| 模块 | 功能 |
|------|------|
| `divide_area.py` | 基于最近机巢原则划分任务区域 |
| `path_planner.py` | 蚁群算法路径规划（含启发式函数） |
| `rotated_test.py` | 旋转牛耕法区域全覆盖路径生成 |
| `energy_calculator.py` | 无人机能耗模型 |
| `tools.py` | 坐标转换、距离计算等通用工具 |

---

## 🗺️ 地图坐标系统

- 地图数据以 **CSV 矩阵** 形式存储
- 每个单元格的值代表该点的高程（单位：米）
- 坐标点以 `(行, 列)` 索引表示
- 支持不同精度地图间的**坐标转换**（缩放+取整）[reference:12]


