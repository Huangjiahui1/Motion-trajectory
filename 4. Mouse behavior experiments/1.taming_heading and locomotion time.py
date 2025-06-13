#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:30:36 2025

@author: hjh
"""

import pandas as pd
import numpy as np
import math
import cv2
import os
'''
test
'''
# 读取动物识别轨迹数据
animal_data = pd.read_csv('/media/hjh/406EB92C6EB91B9A/opencv/output_主动驯服/csvdata/77-246DLC_resnet50_taming1Feb17shuffle1_1030000_filtered.csv',header=2)
# 读取手部识别轨迹数据
hand_data = pd.read_csv('/media/hjh/406EB92C6EB91B9A/opencv/output_主动驯服/handdata/77-246DLC_resnet50_handFeb19shuffle1_1030000_filtered.csv',header=2)


hand_data.columns = ['coords', 'finger1_x', 'finger1_y', 'hand_likelihood', 'finger2_x', 'finger2_y',
       'hand_likelihood.1', 'finger3_x', 'finger3_y', 'hand_likelihood.2',
       'finger4_x', 'finger4_y', 'hand_likelihood.3', 'finger5_x', 'finger5_y',
       'hand_likelihood.4', 'finger6_x', 'finger6_y', 'hand_likelihood.5']
# 修改动物数据列名
animal_data.columns = ['animal_' + col if col != 'coords' else col for col in animal_data.columns]
# 假设两个文件的第一列都是帧索引，根据帧索引合并数据
merged_data = pd.merge(animal_data, hand_data, left_index=True, right_index=True, on='coords')

# 定义函数计算向量夹角
# 定义函数计算向量夹角
# 定义函数计算向量夹角
def calculate_angle(vec1, vec2):
    # 逐元素计算点积
    dot_product = vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1]
    # 逐元素计算向量的范数
    norm_vec1 = np.sqrt(vec1[:, 0] ** 2 + vec1[:, 1] ** 2)
    norm_vec2 = np.sqrt(vec2[:, 0] ** 2 + vec2[:, 1] ** 2)
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    return np.degrees(np.arccos(cos_theta))

# 向量化计算小鼠的朝向向量
def get_orientation_vector_vec(nose_x, nose_y, ear1_x, ear1_y, ear2_x, ear2_y):
    # 计算双耳中点
    mid_ear_x = (ear1_x + ear2_x) / 2
    mid_ear_y = (ear1_y + ear2_y) / 2
    # 计算从双耳中点到鼻尖的向量
    orientation_x = nose_x - mid_ear_x
    orientation_y = nose_y - mid_ear_y
    return np.column_stack([orientation_x, orientation_y])

# 向量化判断点是否在注意力区间内
def is_in_attention_zone_vec(nose_x, nose_y, ear1_x, ear1_y, ear2_x, ear2_y, finger_x, finger_y):
    orientation = get_orientation_vector_vec(nose_x, nose_y, ear1_x, ear1_y, ear2_x, ear2_y)
    nose_to_finger_x = finger_x - nose_x
    nose_to_finger_y = finger_y - nose_y
    nose_to_finger = np.column_stack([nose_to_finger_x, nose_to_finger_y])
    angle = calculate_angle(orientation, nose_to_finger)
    return np.abs(angle) <= 30

# 提取小鼠鼻尖、左耳、右耳的坐标
nose_x = merged_data['animal_x']
nose_y = merged_data['animal_y']
ear1_x = merged_data['animal_x.1']
ear1_y = merged_data['animal_y.1']
ear2_x = merged_data['animal_x.2']
ear2_y = merged_data['animal_y.2']


results = []
for i in range(1, 6):
    finger_x = merged_data[f'finger{i}_x']
    finger_y = merged_data[f'finger{i}_y']
    result = is_in_attention_zone_vec(nose_x, nose_y, ear1_x, ear1_y, ear2_x, ear2_y, finger_x, finger_y)
    results.append(result)

# 只要有一个指尖在注意力区间内就算符合条件
any_finger_in_zone = np.any(results, axis=0)

# 假设视频文件名为 'your_video.mp4'，请根据实际情况修改
cap = cv2.VideoCapture('/media/hjh/406EB92C6EB91B9A/opencv/output_主动驯服/77-246.MP4')

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 获取当前帧的鼻尖、左耳、右耳坐标
    current_nose_x = int(nose_x[frame_count])
    current_nose_y = int(nose_y[frame_count])
    current_ear1_x = int(ear1_x[frame_count])
    current_ear1_y = int(ear1_y[frame_count])
    current_ear2_x = int(ear2_x[frame_count])
    current_ear2_y = int(ear2_y[frame_count])

    # 绘制鼻尖、左耳、右耳的追踪点
    cv2.circle(frame, (current_nose_x, current_nose_y), 5, (0, 255, 0), -1)
    cv2.circle(frame, (current_ear1_x, current_ear1_y), 5, (0, 0, 255), -1)
    cv2.circle(frame, (current_ear2_x, current_ear2_y), 5, (0, 0, 255), -1)

    # 计算注意力区间
    orientation = get_orientation_vector_vec(np.array([current_nose_x]), np.array([current_nose_y]),
                                             np.array([current_ear1_x]), np.array([current_ear1_y]),
                                             np.array([current_ear2_x]), np.array([current_ear2_y]))
    orientation_angle = math.atan2(orientation[0, 1], orientation[0, 0])
    start_angle = math.degrees(orientation_angle - math.radians(30))
    end_angle = math.degrees(orientation_angle + math.radians(30))

    # 绘制注意力区间（半透明）
    overlay = frame.copy()
    cv2.ellipse(overlay, (current_nose_x, current_nose_y), (100, 100), 0, start_angle, end_angle, (255, 0, 0), -1)
    alpha = 0.2  # 透明度
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 判断当前帧是否为 heading 帧
    is_heading = any_finger_in_zone[frame_count]
    text = "Heading" if is_heading else "Not Heading"
    cv2.putText(frame, text, (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示帧
    cv2.imshow('Video', frame)

    # 按 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

'''
batch processing
'''
#####################heading
# 文件夹路径
csvdata_folder = '/media/hjh/406EB92C6EB91B9A/opencv/output_主动驯服/csvdata'
handdata_folder = '/media/hjh/406EB92C6EB91B9A/opencv/output_主动驯服/handdata'

csvdata_folder = '/media/hjh/406EB92C6EB91B9A/opencv/output_被动驯服/csvdata'
handdata_folder = '/media/hjh/406EB92C6EB91B9A/opencv/output_被动驯服/handdata'

# 获取 csvdata 文件夹下的所有 CSV 文件
csvdata_files = [f for f in os.listdir(csvdata_folder) if f.endswith('.csv')]
all_result1 = []
# 遍历每个 CSV 文件
for csvdata_file in csvdata_files:
    # 构建对应的 handdata 文件路径
    handdata_file = csvdata_file.replace('taming1', 'hand').replace('Feb17', 'Feb19')
    handdata_path = os.path.join(handdata_folder, handdata_file)
    csvdata_path = os.path.join(csvdata_folder, csvdata_file)

    # 检查对应的 handdata 文件是否存在
    if os.path.exists(handdata_path):
        # 读取动物识别轨迹数据
        animal_data = pd.read_csv(csvdata_path,header=2)
        # 读取手部识别轨迹数据
        hand_data = pd.read_csv(handdata_path,header=2)

        # 修改动物数据列名
        animal_data.columns = ['animal_' + col if col != 'coords' else col for col in animal_data.columns]
        # 修改手部数据列名
        hand_data.columns = ['coords', 'finger1_x', 'finger1_y', 'hand_likelihood', 'finger2_x', 'finger2_y',
               'hand_likelihood.1', 'finger3_x', 'finger3_y', 'hand_likelihood.2',
               'finger4_x', 'finger4_y', 'hand_likelihood.3', 'finger5_x', 'finger5_y',
               'hand_likelihood.4', 'finger6_x', 'finger6_y', 'hand_likelihood.5']

        # 假设两个文件的第一列都是帧索引，根据帧索引合并数据
        merged_data = pd.merge(animal_data, hand_data, left_index=True, right_index=True, on='coords')

        # 提取小鼠鼻尖、左耳、右耳的坐标
        nose_x = merged_data['animal_x']
        nose_y = merged_data['animal_y']
        ear1_x = merged_data['animal_x.1']
        ear1_y = merged_data['animal_y.1']
        ear2_x = merged_data['animal_x.2']
        ear2_y = merged_data['animal_y.2']

        results = []
        for i in range(1, 6):
            finger_x = merged_data[f'finger{i}_x']
            finger_y = merged_data[f'finger{i}_y']
            result = is_in_attention_zone_vec(nose_x, nose_y, ear1_x, ear1_y, ear2_x, ear2_y, finger_x, finger_y)
            results.append(result)

        # 只要有一个指尖在注意力区间内就算符合条件
        any_finger_in_zone = np.any(results, axis=0)

        # 统计符合条件的帧数
        facing_time = any_finger_in_zone.sum()

        # 每帧时间间隔为 1/60 秒
        frame_interval = 1/60
        total_facing_time = facing_time * frame_interval
        all_result1.append([csvdata_file.split('DLC')[0],total_facing_time])
        print(f"文件 {csvdata_file} 对应的小鼠面朝向人手的时间为: {total_facing_time} 秒")
    else:
        print(f"未找到对应的 handdata 文件: {handdata_file}")
pd.DataFrame(all_result1).to_excel('/media/hjh/406EB92C6EB91B9A/opencv/taming1_result.xlsx')

######locomotion
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 读取CSV文件（假设列名为 'x', 'y', 'timestamp'）
data = pd.read_csv('/media/hjh/406EB92C6EB91B9A/opencv/output_被动驯服/csvdata/1-450DLC_resnet50_taming1Feb17shuffle1_1030000_filtered.csv',header=2)
data['x_avg'] = (data['x'] + data['x.1'] + data['x.2'] + data['x.3'])/4
data['y_avg'] = (data['y'] + data['y.1'] + data['y.2'] + data['y.3'])/4
# 1 pixel = 0.05291 cm
x = data['x'].values*0.05291  # X坐标序列
y = data['y'].values*0.05291  # Y坐标序列
x1 = data['x.1'].values*0.05291  # X坐标序列
y1 = data['y.1'].values*0.05291  # Y坐标序列
x2 = data['x.2'].values*0.05291  # X坐标序列
y2 = data['y.2'].values*0.05291  # Y坐标序列
x3 = data['x.3'].values*0.05291  # X坐标序列
y3 = data['y.3'].values*0.05291  # Y坐标序列

time = data['coords'].values/60  # 时间戳（单位：秒）

# 平滑处理（Savitzky-Golay滤波，窗口15帧，3阶多项式）
window_size = 15  # 需为奇数
x_smooth = savgol_filter(x, window_size, 3)
y_smooth = savgol_filter(y, window_size, 3)
x1_smooth = savgol_filter(x1, window_size, 3)
y1_smooth = savgol_filter(y1, window_size, 3)
x2_smooth = savgol_filter(x1, window_size, 3)
y2_smooth = savgol_filter(y1, window_size, 3)
x3_smooth = savgol_filter(x1, window_size, 3)
y3_smooth = savgol_filter(y1, window_size, 3)

x_avg = (x_smooth + x1_smooth + x2_smooth + x3_smooth)/4
y_avg = (y_smooth + y1_smooth + y2_smooth + y3_smooth)/4

# 计算位移差分
dx = np.gradient(x_avg)  # 中心差分
dy = np.gradient(y_avg)

# 计算时间间隔（假设均匀采样）
dt = np.mean(np.diff(time))  # 平均时间间隔（单位：秒）

# 计算瞬时速度（单位：cm/s，假设坐标已转换为cm）
speed = np.sqrt(dx**2 + dy**2) / dt

# 定义速度阈值（示例：4 cm/s）
speed_threshold = 6

# 生成二值状态序列（1=运动，0=静止）
locomotion_status = (speed >= speed_threshold).astype(int)

# 过滤短暂波动（需持续至少N帧才算有效状态）
min_duration_frames = 20  # 示例：10帧（假设30fps时约0.33秒）
from scipy.ndimage import binary_closing
locomotion_status = binary_closing(locomotion_status, structure=np.ones(min_duration_frames))

# 计算总运动时间（秒）
total_locomotion_time = np.sum(locomotion_status) * dt
print(f"Total locomotion time: {total_locomotion_time:.2f} seconds")

plt.figure(figsize=(12, 4))
plt.plot(time, speed, label='Speed (cm/s)', alpha=0.5)
plt.axhline(speed_threshold, color='r', linestyle='--', label='Threshold')
plt.fill_between(time, 0, 1, where=locomotion_status, 
                 color='g', alpha=0.3, transform=plt.gca().get_xaxis_transform(),
                 label='Locomotion')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.legend()
plt.show()

# 打开视频文件 
video_path = '/media/hjh/406EB92C6EB91B9A/opencv/output_被动驯服/1-450.MP4'  # 替换为实际视频文件路径
cap = cv2.VideoCapture(video_path)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 获取当前帧的速度和运动状态
    current_speed = speed[frame_index]
    current_status = locomotion_status[frame_index]
    status_text = "Moving" if current_status == 1 else "Stationary"

    # 在视频帧的正中间显示速度和运动状态
    text = f"Speed: {current_speed:.2f} cm/s, Status: {status_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)
    
    # 显示视频帧
    cv2.imshow('Video with Speed and Status', frame)
    
    # 等待按键事件，按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_index += 1
# 释放资源
cap.release()
cv2.destroyAllWindows()


















