#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:59:40 2025

@author: hjh
"""

import os
import cv2
import pandas as pd
import numpy as np
# 步骤 1：读取视频文件并获取尺寸
video_path = '/media/hjh/406EB92C6EB91B9A/opencv/output/1/41-215.MP4'  # 替换为实际的视频文件路径
video_path = '/media/hjh/406EB92C6EB91B9A/opencv/output/1/47-255.MP4'  # 替换为实际的视频文件路径
def analyze_video(video_path, trajectory_file):
    # 读取视频文件并获取尺寸
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频的宽度和高度
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 读取视频第一帧作为实拍图片
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        cap.release()
        return
    # 释放视频捕获对象
    cap.release()

    # 划分四个区域
    # 四个区域的像素坐标范围
    regions = [
        # 左上角区域
        ((0, video_width // 2), (0, video_height // 2)),
        # 右上角区域
        ((video_width // 2, video_width), (0, video_height // 2)),
        # 左下角区域
        ((0, video_width // 2), (video_height // 2, video_height)),
        # 右下角区域
        ((video_width // 2, video_width), (video_height // 2, video_height))
    ]

    # 圆形区域参数
    circle_center = (video_width // 2, video_height // 2)
    circle_radius = 27
  # 在图片上标注五个区域
    # 绘制四个方形区域
    for i, ((x_min, x_max), (y_min, y_max)) in enumerate(regions):
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"Region {i + 1}", (x_min + 10, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 绘制圆形区域
    cv2.circle(frame, circle_center, circle_radius, (0, 0, 255), 2)
    cv2.putText(frame, "Circle Region", (circle_center[0] - 50, circle_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 保存标注后的图片
    output_image_path = os.path.splitext(video_path)[0] + "_labeled.jpg"
    cv2.imwrite('/media/hjh/406EB92C6EB91B9A/opencv/photo', frame)
    print(f"已保存标注后的图片: {output_image_path}")

    # 使用 pandas 导入轨迹数据
    try:
        trajectory_df = pd.read_csv(trajectory_file,sep=',',header=2)
    except FileNotFoundError:
        print(f"未找到轨迹数据文件: {trajectory_file}")
        return

    # 定义区域判断函数
    def get_region(x, y):
        # 计算点到圆心的距离
        distance = np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2)
        if distance <= circle_radius:
            return 4  # 圆形区域标记为 4
        for i, ((x_min, x_max), (y_min, y_max)) in enumerate(regions):
            if x_min <= x < x_max and y_min <= y < y_max:
                return i
        return -1  # 如果不在任何区域内

    trajectory_df['avg_x'] = (trajectory_df['x'] + trajectory_df['x.1'] + trajectory_df['x.2'])/3
    trajectory_df['avg_y'] = (trajectory_df['y'] + trajectory_df['y.1'] + trajectory_df['y.2'])/3
    trajectory_df = trajectory_df[['coords','avg_x','avg_y']]

    # 应用区域判断函数
    trajectory_df['region'] = trajectory_df.apply(lambda row: get_region(row['avg_x'], row['avg_y']), axis=1)

    # 计算每个时间间隔
    trajectory_df['time_diff'] = trajectory_df['coords'].diff()

    # 按区域分组并求和时间间隔
    time_in_regions = trajectory_df.groupby('region')['time_diff'].sum().to_dict()

    # 确保每个区域都有结果
    results = []
    for i in range(5):  # 现在有 5 个区域（4 个方形 + 1 个圆形）
        time = time_in_regions.get(i, 0)
        results.append(time)
        if i < 4:
            print(f"视频 {os.path.basename(video_path)} 中，小鼠在区域 {i + 1} 中待的时间为: {time} 单位时间")
        else:
            print(f"视频 {os.path.basename(video_path)} 中，小鼠在圆形过渡区域中待的时间为: {time} 单位时间")
    return results

def process_folder(video_folder, trajectory_folder):
    video_extensions = ['.mp4', '.avi']  # 支持的视频文件扩展名
    all_results = {}

    for filename in os.listdir(video_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in video_extensions:
            base_name = os.path.splitext(filename)[0].replace("_labeled", "")
            video_path = os.path.join(video_folder, filename)
            trajectory_file = os.path.join(trajectory_folder, f"{base_name}.csv")

            print(f"正在处理视频: {filename}")
            results = analyze_video(video_path, trajectory_file)
            if results is not None:
                all_results[filename] = results

    return all_results

# 示例使用
video_folder = '/media/hjh/406EB92C6EB91B9A/opencv/output/2'  # 替换为实际的视频文件夹路径
trajectory_folder = '/media/hjh/406EB92C6EB91B9A/opencv/output/2'  # 替换为实际的轨迹数据文件夹路径
all_results = process_folder(video_folder, trajectory_folder)

# 可以将所有结果保存到 CSV 文件中
result_df = pd.DataFrame.from_dict(all_results, orient='index', columns=[f"区域 {i + 1}" for i in range(4)])
result_df.to_csv('/media/hjh/406EB92C6EB91B9A/opencv/all_analysis_results.csv')



import os
import cv2
import pandas as pd
import numpy as np

# 分析视频函数
def analyze_video(video_path, trajectory_file):
    # 读取视频文件并获取尺寸
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频的宽度和高度
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编解码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.splitext(video_path)[0] + '_analyzed.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (video_width, video_height))

    # 划分四个区域
    regions = [
        # 左上角区域
        ((0, video_width // 2), (0, video_height // 2)),
        # 右上角区域
        ((video_width // 2, video_width), (0, video_height // 2)),
        # 左下角区域
        ((0, video_width // 2), (video_height // 2, video_height)),
        # 右下角区域
        ((video_width // 2, video_width), (video_height // 2, video_height))
    ]

    # 圆形区域参数
    circle_center = (video_width // 2, video_height // 2)
    circle_radius = 27

    # 使用 pandas 导入轨迹数据
    try:
        trajectory_df = pd.read_csv(trajectory_file, sep=',', header=2)
    except FileNotFoundError:
        print(f"未找到轨迹数据文件: {trajectory_file}")
        cap.release()
        out.release()
        return

    trajectory_df['avg_x'] = (trajectory_df['x'] + trajectory_df['x.1'] + trajectory_df['x.2']) / 3
    trajectory_df['avg_y'] = (trajectory_df['y'] + trajectory_df['y.1'] + trajectory_df['y.2']) / 3
    trajectory_df = trajectory_df[['coords', 'avg_x', 'avg_y']]

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 在图片上标注五个区域
        # 绘制四个方形区域
        for i, ((x_min, x_max), (y_min, y_max)) in enumerate(regions):
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Region {i + 1}", (x_min + 10, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制圆形区域
        cv2.circle(frame, circle_center, circle_radius, (0, 0, 255), 2)
        cv2.putText(frame, "Circle Region", (circle_center[0] - 50, circle_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if frame_index < len(trajectory_df):
            x = trajectory_df['avg_x'].iloc[frame_index]
            y = trajectory_df['avg_y'].iloc[frame_index]

            # 计算点到圆心的距离
            distance = np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2)
            if distance <= circle_radius:
                region = 4  # 圆形区域标记为 4
            else:
                for i, ((x_min, x_max), (y_min, y_max)) in enumerate(regions):
                    if x_min <= x < x_max and y_min <= y < y_max:
                        region = i
                        break
                else:
                    region = -1  # 如果不在任何区域内

            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # 绘制标记点
            cv2.putText(frame, f"Current Region: {region + 1 if region >= 0 else 'Outside'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 将处理后的帧写入输出视频
        out.write(frame)

        cv2.imshow('Video Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 处理文件夹函数
def process_folder(video_folder, trajectory_folder):
    video_extensions = ['.MP4', '.avi']  # 支持的视频文件扩展名

    for filename in os.listdir(video_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in video_extensions:
            base_name = os.path.splitext(filename)[0].replace("_labeled", "")
            video_path = os.path.join(video_folder, filename)
            trajectory_file = os.path.join(trajectory_folder, f"{base_name}.csv")

            print(f"正在处理视频: {filename}")
            analyze_video(video_path, trajectory_file)

# 视频文件夹路径
video_folder = r'F:\opencv\output_社交互动2\1'
# 轨迹数据文件夹路径
trajectory_folder = r'F:\opencv\output_社交互动2\1'

process_folder(video_folder, trajectory_folder)



##################################result
import os
import cv2
import pandas as pd
import numpy as np


def analyze_video(video_path, trajectory_file):
    # 读取视频文件并获取尺寸
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频的宽度和高度
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # 使用 pandas 导入轨迹数据
    try:
        trajectory_df = pd.read_csv(trajectory_file, sep=',', header=2)
    except FileNotFoundError:
        print(f"未找到轨迹数据文件: {trajectory_file}")
        return

    # 计算平均坐标
    trajectory_df['avg_x'] = (trajectory_df['x'] + trajectory_df['x.1'] + trajectory_df['x.2']) / 3
    trajectory_df['avg_y'] = (trajectory_df['y'] + trajectory_df['y.1'] + trajectory_df['y.2']) / 3
    trajectory_df = trajectory_df[['coords', 'avg_x', 'avg_y']]

    # 划分四个区域
    regions = [
        # 左上角区域
        ((0, video_width // 2), (0, video_height // 2)),
        # 右上角区域
        ((video_width // 2, video_width), (0, video_height // 2)),
        # 左下角区域
        ((0, video_width // 2), (video_height // 2, video_height)),
        # 右下角区域
        ((video_width // 2, video_width), (video_height // 2, video_height))
    ]

    # 定义区域判断函数
    def get_region(x, y):
        for i, ((x_min, x_max), (y_min, y_max)) in enumerate(regions):
            if x_min <= x < x_max and y_min <= y < y_max:
                return i
        return -1  # 如果不在任何区域内

    # 应用区域判断函数
    trajectory_df['region'] = trajectory_df.apply(lambda row: get_region(row['avg_x'], row['avg_y']), axis=1)

    # 计算每个时间间隔
    trajectory_df['time_diff'] = trajectory_df['coords'].diff()

    # 按区域分组并求和时间间隔
    time_in_regions = trajectory_df.groupby('region')['time_diff'].sum().to_dict()

    # 确保每个区域都有结果
    results = []
    for i in range(4):
        time = time_in_regions.get(i, 0)
        results.append(time)
        print(f"视频 {os.path.basename(video_path)} 中，小鼠在区域 {i + 1} 中待的时间为: {time} 单位时间")
    return results


def process_folder(video_folder, trajectory_folder):
    video_extensions = ['.mp4', '.avi']  # 支持的视频文件扩展名
    all_results = {}

    for filename in os.listdir(video_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in video_extensions:
            base_name = os.path.splitext(filename)[0].replace("_labeled", "")
            video_path = os.path.join(video_folder, filename)
            trajectory_file = os.path.join(trajectory_folder, f"{base_name}.csv")

            print(f"正在处理视频: {filename}")
            results = analyze_video(video_path, trajectory_file)
            if results is not None:
                all_results[filename] = results

    return all_results


# 视频文件夹路径
video_folder = r'F:\opencv\output_社交互动2\1'
# 轨迹数据文件夹路径
trajectory_folder = r'F:\opencv\output_社交互动2\1'

all_results = process_folder(video_folder, trajectory_folder)



    
    
    
    