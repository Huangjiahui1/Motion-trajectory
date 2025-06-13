# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:35:16 2025

@author: h
"""
from moviepy.editor import VideoFileClip
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def input_corners():
    """
    让用户手动输入四个角点的坐标
    """
    corners = []
    print("请依次输入四个角点的坐标（格式：x,y）：")
    for i in range(4):
        while True:
            try:
                corner_str = input(f"请输入第 {i + 1} 个角点的坐标: ")
                x, y = map(int, corner_str.split(','))
                corners.append((x, y))
                break
            except ValueError:
                print("输入格式有误，请输入形如 x,y 的坐标。")
    return np.array(corners, dtype=np.float32)


def process_video(input_path, output_path, selected_corners, start_time, end_time):
    try:
        # 加载视频片段
        clip = VideoFileClip(input_path).subclip(start_time, end_time)

        # 计算裁剪区域，并向外扩展 100 像素
        px = 100
        min_x = int(max(0, np.min(selected_corners[:, 0]) - px))
        max_x = int(np.max(selected_corners[:, 0]) + px)
        min_y = int(max(0, np.min(selected_corners[:, 1]) - px))
        max_y = int(np.max(selected_corners[:, 1]) + px)

        # 调整角点坐标到裁剪区域内
        adjusted_corners = selected_corners - [min_x, min_y]

        # 假设四边形的下边由最后两个角点构成
        p1, p2 = adjusted_corners[2], adjusted_corners[3]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # 进行裁剪
        cropped_clip = clip.crop(x1=min_x, y1=min_y, x2=max_x, y2=max_y)

        # 旋转视频帧，使得四边形下边水平
        rotated_clip = cropped_clip.rotate(angle)

        # 写入处理后的视频，指定比特率为 4000kbps
        bitrate = "4000k"
        rotated_clip.write_videofile(
            output_path, codec='libx264', bitrate=bitrate)

        # 关闭视频文件
        clip.close()
        cropped_clip.close()
        rotated_clip.close()

        print(f"{input_path} 处理完成。")
    except Exception as e:
        print(f"处理视频 {input_path} 时出错: {e}")


def process_video_batch(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(
        input_folder) if f.endswith(('.mp4', '.avi', '.mov','.MP4'))]

    # 存储所有视频的处理参数
    video_params = []
    d = 0
    for file in video_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        print(f"现在处理视频: {input_path}")
        # 手动输入角点
        selected_corners = input_corners()

        while True:
            try:
                start_time = float(input(f"请输入视频 {input_path} 的起始秒: "))
                end_time = float(input(f"请输入视频 {input_path} 的结束秒: "))
                if start_time < 0 or end_time < start_time:
                    print("起始秒不能为负，且结束秒必须大于起始秒，请重新输入。")
                    continue
                break
            except ValueError:
                print("输入的时间格式有误，请输入有效的数字。")

        # 存储当前视频的处理参数
        video_params.append(
            (input_path, output_path, selected_corners, 0, 900))
        d += 1
    # 使用线程池同时处理视频，最多 3 个视频同时运行
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for params in video_params:
            input_path, output_path, selected_corners, start_time, end_time = params
            future = executor.submit(
                process_video, input_path, output_path, selected_corners, start_time, end_time)
            futures.append(future)

        # 等待所有任务完成
        for future in futures:
            future.result()


# 示例用法
input_folder = 'your_input_folder'
output_folder = 'your_output_folder'
process_video_batch(input_folder, output_folder)


# 示例用法
input_folder = 'your_input_folder'
output_folder = 'your_output_folder'
process_video_batch(input_folder, output_folder)
