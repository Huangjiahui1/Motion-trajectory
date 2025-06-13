# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:32:08 2025

@author: h
"""

import cv2
import os

# 定义固定的分割区域
regions = [
    (59, 10, 146, 99),
    (146, 11, 241, 100),
    (60, 99, 148, 188),
    (146, 100, 241, 189)
]

# 处理单个视频的函数
def process_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频的帧率、尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义视频编码器并创建四个视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_videos = [cv2.VideoWriter(f'F:/项目数据/DLC-GWAS文章整理/行为识别正式试验/黄家辉矿场2025.1.17/output/{video_name}_output_{i}.avi', fourcc, fps,
                                  (regions[i][2] - regions[i][0], regions[i][3] - regions[i][1])) for i in
                  range(4)]

    # 逐帧处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 分割帧并保存到四个视频文件中
        for i, (x1, y1, x2, y2) in enumerate(regions):
            cropped_frame = frame[y1:y2, x1:x2]
            out_videos[i].write(cropped_frame)

    # 释放资源
    cap.release()
    for out in out_videos:
        out.release()

# 视频文件夹路径
video_folder = r'F:\项目数据\DLC-GWAS文章整理\行为识别正式试验\黄家辉矿场2025.1.17\矿场视频'

# 遍历文件夹中的所有视频文件
for root, dirs, files in os.walk(video_folder):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov')):  # 支持常见视频格式
            video_path = os.path.join(root, file)
            process_video(video_path)






