#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:33:15 2025

@author: hjh
"""

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import cv2
import pandas as pd
'''
test
'''
def generate_convex_hull(points):
    """生成凸包多边形"""
    hull = ConvexHull(points)
    return points[hull.vertices]

def calculate_overlap(poly1, poly2, area_threshold=0.05):
    """检查重叠面积是否超过阈值"""
    intersection = poly1.intersection(poly2)
    if intersection.is_empty:
        return False
    min_area = min(poly1.area, poly2.area)
    return (intersection.area / min_area) >= area_threshold

# 读取小鼠的轨迹数据
mouse_data = pd.read_csv('F:/opencv/output_主动驯服/csvdata/11-445DLC_resnet50_taming1Feb17shuffle2_1030000_filtered.csv',header=2) # 替换为实际的小鼠轨迹数据CSV文件路径
mouse_data.columns = ['coords', 'x.0', 'y.0', 'likelihood.0', 'x.1', 'y.1', 'likelihood.1', 'x.2',
       'y.2', 'likelihood.2', 'x.3', 'y.3', 'likelihood.3', 'x.4', 'y.4',
       'likelihood.4', 'x.5', 'y.5', 'likelihood.5', 'x.6', 'y.6',
       'likelihood.6', 'x.7', 'y.7', 'likelihood.7']
num_mouse_frames = len(mouse_data)
mouse_trajs = np.zeros((num_mouse_frames, 8, 2))
for i in range(num_mouse_frames):
    for j in range(8):
        mouse_trajs[i, j, 0] = mouse_data.iloc[i][f'x.{j}']
        mouse_trajs[i, j, 1] = mouse_data.iloc[i][f'y.{j}']

# 读取手部的轨迹数据
hand_data = pd.read_csv('F:/opencv/output_主动驯服/handdata/11-445DLC_resnet50_handFeb19shuffle1_1030000_filtered.csv',header=2)  # 替换为实际的手部轨迹数据CSV文件路径
hand_data.columns = ['coords', 'x.0', 'y.0', 'likelihood.0', 'x.1', 'y.1', 'likelihood.1', 'x.2',
       'y.2', 'likelihood.2', 'x.3', 'y.3', 'likelihood.3', 'x.4', 'y.4',
       'likelihood.4', 'x.5', 'y.','likelihood.5']
num_hand_frames = len(hand_data)
hand_trajs = np.zeros((num_hand_frames, 5, 2))
for i in range(num_hand_frames):
    for j in range(5):
        hand_trajs[i, j, 0] = hand_data.iloc[i][f'x.{j}']
        hand_trajs[i, j, 1] = hand_data.iloc[i][f'y.{j}']

# 确保两个轨迹数据的帧数一致，这里简单假设帧数相同
assert num_mouse_frames == num_hand_frames, "小鼠和手部轨迹数据帧数不一致"

# 参数设置
DISTANCE_THRESHOLD = 15  # 实际距离单位（需根据实验校准）
OVERLAP_THRESHOLD = 0.05
FRAME_RATE = 60
MIN_TOUCHING_DURATION_SECONDS = 0.5  # 最小触碰持续时间，单位：秒
MIN_TOUCHING_FRAMES = int(MIN_TOUCHING_DURATION_SECONDS * FRAME_RATE)  # 换算成帧数

contact_frames = []

# 打开基础实拍视频
cap = cv2.VideoCapture('F:/opencv/output/11-445.MP4')  # 替换为实际的基础实拍视频路径
# 打开视频写入器
out = cv2.VideoWriter('F:/opencv/11-445touch_output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (int(cap.get(3)), int(cap.get(4))))

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 生成手部和小鼠轮廓
    hand_points = hand_trajs[frame_index]
    mouse_points = mouse_trajs[frame_index]
    
    # 跳过无效帧（如缺失关键点）
    if np.any(np.isnan(hand_points)) or np.any(np.isnan(mouse_points)):
        contact_frames.append(False)
        cv2.putText(frame, "Invalid Frame", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        hand_hull = generate_convex_hull(hand_points)
        mouse_hull = generate_convex_hull(mouse_points)
        
        # 创建Shapely多边形对象
        hand_poly = Polygon(hand_hull)
        mouse_poly = Polygon(mouse_hull)
        
        # 条件1：最小距离检查
        min_dist = hand_poly.distance(mouse_poly)
        distance_contact = (min_dist <= DISTANCE_THRESHOLD)
        
        # 条件2：重叠面积检查
        overlap_contact = calculate_overlap(hand_poly, mouse_poly, OVERLAP_THRESHOLD)
        
        contact_frames.append(distance_contact or overlap_contact)

        # 绘制手部和小鼠轮廓
        for i in range(len(hand_hull)):
            cv2.line(frame, tuple(hand_hull[i].astype(int)), tuple(hand_hull[(i+1) % len(hand_hull)].astype(int)), (255, 0, 0), 2)
        for i in range(len(mouse_hull)):
            cv2.line(frame, tuple(mouse_hull[i].astype(int)), tuple(mouse_hull[(i+1) % len(mouse_hull)].astype(int)), (0, 0, 255), 2)

        # 标注接触状态
        contact_text = "Contact" if contact_frames[-1] else "No Contact"
        cv2.putText(frame, contact_text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if contact_frames[-1] else (0, 0, 255), 2)

    # 写入视频帧
    out.write(frame)
    frame_index += 1
# 去噪处理，确保连续接触帧超过最小触碰帧数
contact_array = np.array(contact_frames)
filtered_contact = []
i = 0
while i < len(contact_array):
    if contact_array[i]:
        start = i
        while i < len(contact_array) and contact_array[i]:
            i += 1
        end = i
        if end - start >= MIN_TOUCHING_FRAMES:
            filtered_contact.extend([True] * (end - start))
        else:
            filtered_contact.extend([False] * (end - start))
    else:
        filtered_contact.append(False)
        i += 1

# 计算总时间
total_time = np.sum(filtered_contact) / FRAME_RATE
print(f"Total touching time: {total_time:.2f} seconds")

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

###############################################
'''
batch processing
'''
import os
import numpy as np
from shapely.geometry import Polygon
import pandas as pd

def generate_convex_hull(points):
    if len(points) < 3:  # 对于二维情况，至少需要 3 个点
        return None
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        return None

def calculate_overlap(poly1, poly2, area_threshold=0.05):
    """检查重叠面积是否超过阈值"""
    intersection = poly1.intersection(poly2)
    if intersection.is_empty:
        return False
    min_area = min(poly1.area, poly2.area)
    return (intersection.area / min_area) >= area_threshold

# 定义文件夹路径
hand_folder = 'F:/opencv/output_主动驯服/handdata'
mouse_folder = 'F:/opencv/output_主动驯服/csvdata'

hand_folder = 'F:/opencv/output_被动驯服/handdata'
mouse_folder = 'F:/opencv/output_被动驯服/csvdata'
# 参数设置
DISTANCE_THRESHOLD = 15  # 实际距离单位（需根据实验校准）
OVERLAP_THRESHOLD = 0.05
FRAME_RATE = 60
MIN_TOUCHING_DURATION_SECONDS = 0.5  # 最小触碰持续时间，单位：秒
MIN_TOUCHING_FRAMES = int(MIN_TOUCHING_DURATION_SECONDS * FRAME_RATE)  # 换算成帧数

# 获取文件夹中所有 CSV 文件
hand_files = [f for f in os.listdir(hand_folder) if f.endswith('.csv')]
mouse_files = [f for f in os.listdir(mouse_folder) if f.endswith('.csv')]

# 按文件名中“DLC”之前的部分匹配文件
hand_files.sort(key=lambda x: x.split('DLC')[0])
mouse_files.sort(key=lambda x: x.split('DLC')[0])

# 确保文件数量一致
assert len(hand_files) == len(mouse_files), "手部和小鼠 CSV 文件数量不一致"

# 存储结果的列表
results = []
k = True
for hand_file, mouse_file in zip(hand_files, mouse_files):
    
    #if hand_file == '118-222DLC_resnet50_handFeb19shuffle1_1030000_filtered.csv':
    #    k = False
    #if k :
    #    continue
    # 读取小鼠的轨迹数据
    mouse_data = pd.read_csv(os.path.join(mouse_folder, mouse_file),header=2)
    num_mouse_frames = len(mouse_data)
    mouse_data.columns = ['coords', 'x.0', 'y.0', 'likelihood.0', 'x.1', 'y.1', 'likelihood.1', 'x.2',
           'y.2', 'likelihood.2', 'x.3', 'y.3', 'likelihood.3', 'x.4', 'y.4',
           'likelihood.4', 'x.5', 'y.5', 'likelihood.5', 'x.6', 'y.6',
           'likelihood.6', 'x.7', 'y.7', 'likelihood.7']
    
    mouse_trajs = np.array([mouse_data[[f'x.{j}', f'y.{j}']].values for j in range(8)]).transpose(1, 0, 2)

    # 读取手部的轨迹数据
    hand_data = pd.read_csv(os.path.join(hand_folder, hand_file),header=2)
    num_hand_frames = len(hand_data)
    hand_data.columns = ['coords', 'x.0', 'y.0', 'likelihood.0', 'x.1', 'y.1', 'likelihood.1', 'x.2',
           'y.2', 'likelihood.2', 'x.3', 'y.3', 'likelihood.3', 'x.4', 'y.4',
           'likelihood.4', 'x.5', 'y.','likelihood.5']

    if num_mouse_frames - num_hand_frames == -1 :
        hand_data = hand_data.iloc[:-1]
    elif num_mouse_frames - num_hand_frames == 1:
        mouse_data = mouse_data.iloc[:-1]
    
    num_mouse_frames = len(mouse_data)
    num_hand_frames = len(hand_data)
    hand_trajs = np.array([hand_data[[f'x.{j}', f'y.{j}']].values for j in range(5)]).transpose(1, 0, 2)
    # 确保两个轨迹数据的帧数一致
    assert num_mouse_frames == num_hand_frames, f"{mouse_file} 和 {hand_file} 的帧数不一致"

    contact_frames = []

    for frame in range(len(hand_trajs)):
        # 生成手部和小鼠轮廓
        hand_points = hand_trajs[frame]
        mouse_points = mouse_trajs[frame]

        # 跳过无效帧（如缺失关键点）
        if np.any(np.isnan(hand_points)) or np.any(np.isnan(mouse_points)):
            contact_frames.append(False)
            continue

        hand_hull = generate_convex_hull(hand_points)
        mouse_hull = generate_convex_hull(mouse_points)

        # 创建 Shapely 多边形对象
        hand_poly = Polygon(hand_hull)
        mouse_poly = Polygon(mouse_hull)

        # 条件 1：最小距离检查
        min_dist = hand_poly.distance(mouse_poly)
        distance_contact = (min_dist <= DISTANCE_THRESHOLD)

        # 条件 2：重叠面积检查
        overlap_contact = calculate_overlap(hand_poly, mouse_poly, OVERLAP_THRESHOLD)

        contact_frames.append(distance_contact or overlap_contact)

    # 去噪处理，确保连续接触帧超过最小触碰帧数
    contact_array = np.array(contact_frames)
    filtered_contact = []
    i = 0
    while i < len(contact_array):
        if contact_array[i]:
            start = i
            while i < len(contact_array) and contact_array[i]:
                i += 1
            end = i
            if end - start >= MIN_TOUCHING_FRAMES:
                filtered_contact.extend([True] * (end - start))
            else:
                filtered_contact.extend([False] * (end - start))
        else:
            filtered_contact.append(False)
            i += 1

    # 计算总时间
    total_time = np.sum(filtered_contact) / FRAME_RATE
    # 存储文件名和总时间
    name = mouse_file.split('DLC')[0]
    print([name, total_time])
    results.append([name, total_time])

# 将结果保存到 CSV 文件
result_df = pd.DataFrame(results, columns=['name', 'time'])
result_df.to_csv('F:\opencv\contact_time_10cm驯服_results.csv', index=False)
print("结果已保存到 contact_time_results.csv 文件中。")







