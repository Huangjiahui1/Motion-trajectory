#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:44:13 2025

@author: hjh
"""

import pandas as pd

# 读取CSV文件
csv_path = '/media/hjh/406EB92C6EB91B9A/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/对照+2025年1月17日第1次测试+对照-001+对照-002+对照-003+对照-004+4_output_0DLC_resnet50_open_fieldFeb28shuffle1_150000_filtered.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_path, header=2)

tag = csv_path.split('output_')[1][0]
dict1 = {'0':[86,88],'1':[94,88],'2':[88,88],'3':[94,88]}

# 视频帧的尺寸（需要根据实际情况调整）
video_width = dict1[tag][0]  # 视频宽度
video_height = dict1[tag][1]  # 视频高度



# 划分网格
grid_size = 5  # 5x5网格
grid_width = video_width // grid_size
grid_height = video_height // grid_size

# 中心区域的网格索引（中间9格）
center_grid = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

# 初始化时间统计
time_in_center = 0
time_in_periphery = 0

# 遍历轨迹点
for index, row in df.iterrows():
    x, y = row['x'], row['y']  # 获取小鼠的坐标

    # 判断所在的网格
    grid_x = x // grid_width
    grid_y = y // grid_height

    # 判断是否在中心区域
    if (grid_x, grid_y) in center_grid:
        time_in_center += 1
    else:
        time_in_periphery += 1

# 假设视频的帧率（需要根据实际情况调整）
fps = 60  # 视频帧率

# 计算时间（单位：秒）
time_in_center /= fps
time_in_periphery /= fps

print(f"小鼠在中心区域的时间: {time_in_center:.2f} 秒")
print(f"小鼠在外围区域的时间: {time_in_periphery:.2f} 秒")


import os
import pandas as pd

import numpy as np
mouse_folder = 'F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'
# 获取文件夹中所有 CSV 文件
mouse_folder = [f for f in os.listdir(mouse_folder) if f.endswith('.csv')]

# 读取CSV文件
#csv_path = 'F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/全敲+2025年1月18日第1次测试+全敲-001+全敲-002+全敲-003+全敲-004+4_output_0DLC_resnet50_open_fieldFeb28shuffle1_150000_filtered.csv'
results = []
for csv_path in mouse_folder:
    df = pd.read_csv('F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'+csv_path, header=2)
    name = csv_path.split('/')[-1]
    # 从文件名中提取视频尺寸信息
    tag = csv_path.split('output_')[1][0]
    dict1 = {'0': [86, 88], '1': [94, 88], '2': [88, 88], '3': [94, 88]}
    df.x = df.x - 3
    # 视频帧的尺寸
    video_width, video_height = dict1[tag]
    
    # 划分网格
    grid_size = 5  # 5x5网格
    grid_width = video_width // grid_size
    grid_height = video_height // grid_size
    
    # 计算剩余像素并调整起始位置
    remaining_width = video_width % grid_size
    remaining_height = video_height % grid_size
    
    # 起始位置偏移（将剩余像素均匀分配到两侧）
    start_x = remaining_width // 2
    start_y = remaining_height // 2
    # 中心区域的网格索引（中间9格）
    center_grid = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    
    # 初始化时间统计
    time_in_center = 0
    time_in_periphery = 0
    
    # 遍历轨迹点
    for index, row in df.iterrows():
        x, y = row['x'], row['y']  # 获取小鼠的坐标
    
        # 判断是否在有效区域内（忽略剩余像素）
        if x < start_x or x >= start_x + grid_width * grid_size or \
           y < start_y or y >= start_y + grid_height * grid_size:
            continue  # 超出有效区域，忽略该点
    
        # 计算所在的网格
        grid_x = (x - start_x) // grid_width
        grid_y = (y - start_y) // grid_height
    
        # 判断是否在中心区域
        if (grid_x, grid_y) in center_grid:
            time_in_center += 1
        else:
            time_in_periphery += 1
    
    # 假设视频的帧率（需要根据实际情况调整）
    fps = 20  # 视频帧率
    
    # 计算时间（单位：秒）
    time_in_center /= fps
    time_in_periphery /= fps
    
    print(f"小鼠在中心区域的时间: {time_in_center:.2f} 秒")
    print(f"小鼠在外围区域的时间: {time_in_periphery:.2f} 秒")
    
    
    import matplotlib.pyplot as plt
    
    # 绘制网格
    for i in range(grid_size + 1):
        plt.axvline(x=start_x + i * grid_width, color='red', linestyle='--')
        plt.axhline(y=start_y + i * grid_height, color='red', linestyle='--')
    
    # 绘制小鼠轨迹
    plt.scatter(df['x'], df['y'], c='blue', label='小鼠轨迹')
    
    # 设置画面边界
    plt.xlim(0, video_width)
    plt.ylim(0, video_height)
    plt.gca().invert_yaxis()  # 翻转Y轴
    plt.legend()
    plt.xlabel(name.split('+')[int(name.split('+')[-1].split('DLC')[0][-1])+2])
    plt.ylabel('Y坐标 (像素)')
    plt.show()

    df = pd.read_csv('F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'+csv_path, header=2)

    # 假设df中有'x'和'y'两列表示小鼠的坐标
    # 假设df中还有'likelihood'列表示置信度

    # 设置置信度阈值
    likelihood_threshold = 0.9

    # 剔除置信度低于阈值的点
    df = df[df['likelihood'] >= likelihood_threshold]

    # 使用移动平均滤波平滑轨迹
    window_size = 5  # 滤波窗口大小
    df['x_smooth'] = df['x'].rolling(window=window_size, center=True).mean()
    df['y_smooth'] = df['y'].rolling(window=window_size, center=True).mean()

    # 填充滤波后的NaN值
    df['x_smooth'] = df['x_smooth'].fillna(method='ffill').fillna(method='bfill')
    df['y_smooth'] = df['y_smooth'].fillna(method='ffill').fillna(method='bfill')

    # 初始化总路程
    total_distance = 0

    # 提取滤波后的坐标
    x_smooth = df['x_smooth'].values
    y_smooth = df['y_smooth'].values

    # 计算每两个连续点之间的距离
    dx = x_smooth[1:] - x_smooth[:-1]
    dy = y_smooth[1:] - y_smooth[:-1]
    distances = np.sqrt(dx**2 + dy**2)

    # 计算总路程
    total_distance = np.sum(distances)
    results.append([name,time_in_center,total_distance])



# 读取CSV文件
csv_path = 'your_csv_file.csv'
df = pd.read_csv('F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'+csv_path, header=2)

# 假设df中有'x'和'y'两列表示小鼠的坐标
# 假设df中还有'likelihood'列表示置信度

# 设置置信度阈值
likelihood_threshold = 0.9

# 剔除置信度低于阈值的点
df = df[df['likelihood'] >= likelihood_threshold]

# 使用移动平均滤波平滑轨迹
window_size = 5  # 滤波窗口大小
df['x_smooth'] = df['x'].rolling(window=window_size, center=True).mean()
df['y_smooth'] = df['y'].rolling(window=window_size, center=True).mean()

# 填充滤波后的NaN值
df['x_smooth'] = df['x_smooth'].fillna(method='ffill').fillna(method='bfill')
df['y_smooth'] = df['y_smooth'].fillna(method='ffill').fillna(method='bfill')

# 初始化总路程
total_distance = 0

# 提取滤波后的坐标
x_smooth = df['x_smooth'].values
y_smooth = df['y_smooth'].values

# 计算每两个连续点之间的距离
dx = x_smooth[1:] - x_smooth[:-1]
dy = y_smooth[1:] - y_smooth[:-1]
distances = np.sqrt(dx**2 + dy**2)

# 计算总路程
total_distance = np.sum(distances)
# 绘制网格和轨迹
plt.figure(figsize=(8, 8))

# 绘制网格
for i in range(grid_size + 1):
    plt.axvline(x=start_x + i * grid_width, color='red', linestyle='--')
    plt.axhline(y=start_y + i * grid_height, color='red', linestyle='--')

# 绘制小鼠轨迹
plt.plot(df['x'], df['y'], c='blue', label='小鼠轨迹')

# 设置画面边界
plt.xlim(0, video_width)
plt.ylim(0, video_height)
plt.gca().invert_yaxis()  # 翻转Y轴
plt.legend()
plt.xlabel(name.split('+')[int(name.split('+')[-1].split('DLC')[0][-1])+2])
plt.ylabel('Y坐标 (像素)')
plt.title('小鼠轨迹与网格划分')
plt.show()

print(f"小鼠的运动总路程: {total_distance:.2f} 像素")



###############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 指定目录路径
directory_path = 'F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'

# 获取目录下所有CSV文件
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

results = []

for csv_file in csv_files:
    csv_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(csv_path, header=2)
    name = csv_file
    
    # 从文件名中提取视频尺寸信息
    tag = csv_file.split('output_')[1][0]
    dict1 = {'0': [86, 88], '1': [94, 88], '2': [88, 88], '3': [94, 88]}
    df.x = df.x - 3
    
    # 视频帧的尺寸
    video_width, video_height = dict1[tag]
    
    # 划分网格
    grid_size = 5  # 5x5网格
    grid_width = video_width // grid_size
    grid_height = video_height // grid_size
    
    # 计算剩余像素并调整起始位置
    remaining_width = video_width % grid_size
    remaining_height = video_height % grid_size
    
    # 起始位置偏移（将剩余像素均匀分配到两侧）
    start_x = remaining_width // 2
    start_y = remaining_height // 2
    
    # 中心区域的网格索引（中间9格）
    center_grid = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    
    # 设置置信度过滤阈值
    likelihood_threshold = 0.9
    df = df[df['likelihood'] >= likelihood_threshold]
    
    # 使用移动平均滤波平滑轨迹
    window_size = 5  # 滤波窗口大小
    df['x_smooth'] = df['x'].rolling(window=window_size, center=True).mean()
    df['y_smooth'] = df['y'].rolling(window=window_size, center=True).mean()
    
    # 填充滤波后的NaN值
    df['x_smooth'] = df['x_smooth'].fillna(method='ffill').fillna(method='bfill')
    df['y_smooth'] = df['y_smooth'].fillna(method='ffill').fillna(method='bfill')
    
    # 初始化时间统计和总路程
    time_in_center = 0
    total_distance_in_center = 0
    
    # 提取滤波后的坐标
    x_smooth = df['x_smooth'].values
    y_smooth = df['y_smooth'].values
    
    # 遍历轨迹点
    for i in range(1, len(x_smooth)):
        x1, y1 = x_smooth[i - 1], y_smooth[i - 1]
        x2, y2 = x_smooth[i], y_smooth[i]
        
        # 判断点是否在中心区域
        grid_x1 = (x1 - start_x) // grid_width
        grid_y1 = (y1 - start_y) // grid_height
        grid_x2 = (x2 - start_x) // grid_width
        grid_y2 = (y2 - start_y) // grid_height
        
        if (grid_x1, grid_y1) in center_grid:
            time_in_center += 1
            if (grid_x2, grid_y2) in center_grid:
                # 计算两点之间的距离
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                total_distance_in_center += distance
    
    # 假设视频的帧率（需要根据实际情况调整）
    fps = 20  # 视频帧率
    
    # 计算时间（单位：秒）
    time_in_center /= fps
    
    print(f"小鼠在中心区域的时间: {time_in_center:.2f} 秒")
    print(f"小鼠在中心区域的总路程: {total_distance_in_center:.2f} 像素")
    
    results.append([name, time_in_center, total_distance_in_center])

# 如果需要将结果保存到CSV文件
result_df = pd.DataFrame(results, columns=['File Name', 'Time in Center (s)', 'Distance in Center (pixels)'])
result_df.to_csv(os.path.join(directory_path, 'analysis_results.csv'), index=False)
print("分析结果已保存到 'analysis_results.csv' 文件中。")

#########################################

import pandas as pd
import numpy as np
import os

# 指定目录路径
directory_path = 'F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'

# 获取目录下所有CSV文件
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

results = []

# 假设视频的帧率（需要根据实际情况调整）
fps = 20  # 视频帧率

# 比例尺：50cm = 90像素
scale_cm_per_pixel = 50 / 90  # 每像素对应的厘米数
min_speed_threshold_cm_per_s = 3.0  # 最小运动速度阈值（单位：cm/s）
min_speed_threshold_pixel_per_frame = min_speed_threshold_cm_per_s / (fps * scale_cm_per_pixel)

# 中心区域持续时间阈值
min_duration_in_center = 0.5  # 单位：秒
min_frames_in_center = int(min_duration_in_center * fps)  # 对应的帧数

# 连续运动的最小帧数
min_consecutive_motion_frames = 5

for csv_file in csv_files:
    csv_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(csv_path, header=2)
    name = csv_file
    
    # 从文件名中提取视频尺寸信息
    tag = csv_file.split('output_')[1][0]
    dict1 = {'0': [86, 88], '1': [94, 88], '2': [88, 88], '3': [94, 88]}
    df.x = df.x - 3
    
    # 视频帧的尺寸
    video_width, video_height = dict1[tag]
    
    # 划分网格
    grid_size = 5  # 5x5网格
    grid_width = video_width // grid_size
    grid_height = video_height // grid_size
    
    # 计算剩余像素并调整起始位置
    remaining_width = video_width % grid_size
    remaining_height = video_height % grid_size
    
    # 起始位置偏移（将剩余像素均匀分配到两侧）
    start_x = remaining_width // 2
    start_y = remaining_height // 2
    
    # 中心区域的网格索引（中间9格）
    center_grid = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    
    # 设置置信度过滤阈值
    likelihood_threshold = 0.9
    df = df[df['likelihood'] >= likelihood_threshold]
    
    # 使用移动平均滤波平滑轨迹
    window_size = 5  # 滤波窗口大小
    df['x_smooth'] = df['x'].rolling(window=window_size, center=True).mean()
    df['y_smooth'] = df['y'].rolling(window=window_size, center=True).mean()
    
    # 填充滤波后的NaN值
    df['x_smooth'] = df['x_smooth'].fillna(method='ffill').fillna(method='bfill')
    df['y_smooth'] = df['y_smooth'].fillna(method='ffill').fillna(method='bfill')
    
    # 初始化时间统计和总路程
    time_in_center = 0
    total_distance_in_center = 0
    consecutive_frames_in_center = 0  # 连续在中心区域的帧数
    consecutive_motion_frames = 0  # 连续运动的帧数
    is_in_center = False  # 是否处于中心区域
    is_moving = False  # 是否处于运动状态
    prev_x, prev_y = None, None  # 上一帧的坐标
    
    # 提取滤波后的坐标
    x_smooth = df['x_smooth'].values
    y_smooth = df['y_smooth'].values
    
    # 遍历轨迹点
    for i in range(len(x_smooth)):
        x, y = x_smooth[i], y_smooth[i]
        
        # 判断点是否在中心区域
        grid_x = (x - start_x) // grid_width
        grid_y = (y - start_y) // grid_height
        
        if (grid_x, grid_y) in center_grid:
            if not is_in_center:
                # 刚进入中心区域，重置计数器
                consecutive_frames_in_center = 1
                is_in_center = True
            else:
                # 已在中心区域，增加计数器
                consecutive_frames_in_center += 1
        else:
            # 不在中心区域，重置状态
            is_in_center = False
            consecutive_frames_in_center = 0
            consecutive_motion_frames = 0  # 重置连续运动帧数
            is_moving = False  # 重置运动状态
        
        # 检查是否满足持续时间条件
        if consecutive_frames_in_center >= min_frames_in_center:
            if prev_x is not None and prev_y is not None:
                # 计算速度
                distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                speed_pixel_per_frame = distance
                
                if speed_pixel_per_frame >= min_speed_threshold_pixel_per_frame:
                    # 速度超过阈值，认为处于运动状态
                    if not is_moving:
                        # 刚开始运动，重置连续运动帧数
                        consecutive_motion_frames = 1
                        is_moving = True
                    else:
                        # 已在运动状态，增加连续运动帧数
                        consecutive_motion_frames += 1
                else:
                    # 速度未超过阈值，重置运动状态
                    is_moving = False
                    consecutive_motion_frames = 0
            
            # 如果连续运动帧数达到阈值，计入路程
            if consecutive_motion_frames >= min_consecutive_motion_frames:
                if prev_x is not None and prev_y is not None:
                    distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    total_distance_in_center += distance
                    time_in_center += 1  # 增加在中心区域的时间
            
            prev_x, prev_y = x, y
    
    # 将时间转换为秒
    time_in_center /= fps
    
    print(f"小鼠在中心区域的时间: {time_in_center:.2f} 秒")
    print(f"小鼠在中心区域的总路程: {total_distance_in_center:.2f} 像素")
    
    results.append([name, time_in_center, total_distance_in_center])

# 如果需要将结果保存到CSV文件
result_df = pd.DataFrame(results, columns=['File Name', 'Time in Center (s)', 'Distance in Center (pixels)'])
result_df.to_csv(os.path.join(directory_path, 'analysis_results.csv'), index=False)
print("分析结果已保存到 'analysis_results.csv' 文件中。")


#########################################

import pandas as pd
import numpy as np
import os

# 指定目录路径
directory_path = 'F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/csv/'

# 获取目录下所有CSV文件
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

results = []

# 假设视频的帧率（需要根据实际情况调整）
fps = 20  # 视频帧率

# 中心区域持续时间阈值
min_duration_in_center = 0.5  # 单位：秒
min_frames_in_center = int(min_duration_in_center * fps)  # 对应的帧数

# 连续运动的最小帧数
min_consecutive_motion_frames = 5

for csv_file in csv_files:
    csv_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(csv_path, header=2)
    name = csv_file
    
    # 从文件名中提取视频尺寸信息
    tag = csv_file.split('output_')[1][0]
    dict1 = {'0': [86, 88], '1': [94, 88], '2': [88, 88], '3': [94, 88]}
    df.x = df.x - 3
    
    # 视频帧的尺寸
    video_width, video_height = dict1[tag]
    
    # 计算旷场中心点坐标
    center_x = video_width / 2
    center_y = video_height / 2
    
    # 设置置信度过滤阈值
    likelihood_threshold = 0.9
    df = df[df['likelihood'] >= likelihood_threshold]
    
    # 使用移动平均滤波平滑轨迹
    window_size = 5  # 滤波窗口大小
    df['x_smooth'] = df['x'].rolling(window=window_size, center=True).mean()
    df['y_smooth'] = df['y'].rolling(window=window_size, center=True).mean()
    
    # 填充滤波后的NaN值
    df['x_smooth'] = df['x_smooth'].fillna(method='ffill').fillna(method='bfill')
    df['y_smooth'] = df['y_smooth'].fillna(method='ffill').fillna(method='bfill')
    
    # 提取滤波后的坐标
    x_smooth = df['x_smooth'].values
    y_smooth = df['y_smooth'].values
    
    # 初始化距离列表
    distances_to_center = []
    
    # 遍历轨迹点
    for i in range(len(x_smooth)):
        x, y = x_smooth[i], y_smooth[i]
        
        # 计算小鼠到旷场中心点的距离
        distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        distances_to_center.append(distance_to_center)
    
    # 计算平均距离
    average_distance_to_center = np.mean(distances_to_center)
    
    print(f"小鼠到旷场中心点的平均距离: {average_distance_to_center:.2f} 像素")
    
    results.append([name, average_distance_to_center])

# 将结果保存到DataFrame
results_df = pd.DataFrame(results, columns=['File Name', 'Average Distance to Center (pixels)'])
results_df.to_csv(r'F:\data\average_distances_to_center.csv', index=False)
















