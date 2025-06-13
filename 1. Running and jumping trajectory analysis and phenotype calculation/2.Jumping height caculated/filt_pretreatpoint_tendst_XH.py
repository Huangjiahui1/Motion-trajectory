#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:35:01 2021

@author: hjh
"""
import os 
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from Change import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error as MSE,r2_score
from sklearn.svm import OneClassSVM
import pointspin as spin
'''
————————————————————————————————————————————————————————————
两种基于机器学习的过滤器 
————————————————————————————————————————————————————————————
'''  
def LOF(new, n_neighbors=5):
    new = np.array(new)
    x = new[:,1]
    y = new[:,2]
    df = []
    for i in range(len(x)):
        df.append([x[i],y[i]])
    df = np.array(df)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)  
    ypre1 = clf.fit_predict(df)
    toal_data = np.hstack((df, ypre1.reshape(df.shape[0], 1)))
    new_data = np.hstack((new, ypre1.reshape(df.shape[0], 1)))
    normal_new_data =  new_data[new_data[:, -1] == 1]  
    new_data = normal_new_data[:,:-1]
    return new_data

def OCSVM(new, nu = 0.1):
    '''
    输入需要判断的数据x,y
    nu：预计的离群数据量所占百分比
    '''
    new = np.array(new)
    x = new[:,1]
    y = new[:,2]
    df = []
    for i in range(len(x)):
        df.append([x[i],y[i]])
    df = np.array(df)
    #用SVM进行异常值检测
    model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = nu).fit(df)
    ypre = model.predict(df)
    #合并x,y数据和预测结果
    toal_data = np.hstack((df, ypre.reshape(df.shape[0], 1)))
    new_data = np.hstack((new, ypre.reshape(df.shape[0], 1)))
    #得到正常的数据结果
    normal_new_data =  new_data[new_data[:, -1] == 1]
    new_data = normal_new_data[:,:-1]
    return new_data
'''
————————————————————————————————————————————————————————————
csv文件预处理
————————————————————————————————————————————————————————————
'''  

def filt(point):
    #过滤和格式转换
    new = []
    for i in point:  
        if (i[3] == '') or (i[1] == '') or (i[2] == ''):   #or (float(i[2]) < 400)
            continue
        else:
            new.append(i)
    new1 = []
    for news in new:
        news = list(map(float, news))
        new1.append(news)
    new = new1
    #过滤低likelihood的识别点
    new2 = []
    for i in range(len(new)):
        if float(new[i][3]) >= 0.8:
            new2.append(new[i])
    #new = LOF(new2)
    new = new2
    #new=OCSVM(new2, nu = 0.1)
    
    return  new

def safe_float_convert(x):
    try:
        return float(x)
    except ValueError:
        return None  # 返回 None 或者可以是 0，取决于如何处理无效或缺失的数据

def pretreatpoint(path,individual=1):

    '''
    individual:1,2
    '''

    try:
        data = pd.read_csv(path,keep_default_na=False,usecols = range(22))

        #把DataFrame转为list
        datalist = data.values.tolist()
        #分割点的轨迹
        i1p1 = [];i1p2 = [];i2p1 = [];i2p2 = []
        for i in range(len(datalist)):
            i1p1.append([datalist[i][0],datalist[i][1],datalist[i][2],datalist[i][3]])
            i1p2.append([datalist[i][0],datalist[i][4],datalist[i][5],datalist[i][6]])
            i2p1.append([datalist[i][0],datalist[i][13],datalist[i][14],datalist[i][15]])
            i2p2.append([datalist[i][0],datalist[i][16],datalist[i][17],datalist[i][18]])
        i1p1 = i1p1[3:];i1p2 = i1p2[3:];i2p1 = i2p1[3:];i2p2 = i2p2[3:]
        new1 = filt(i1p2)
        new2 = filt(i2p2) 
    except ValueError:        
        if individual==2:
            raise ValueError
            
        data = pd.read_csv(path,keep_default_na=False,usecols = range(13))

        datalist = data.values.tolist()
        i1p1 = [];i1p2 = [];i1p3 = [];i1p4 = []
        for i in range(len(datalist)):
            i1p1.append([datalist[i][0],datalist[i][1],datalist[i][2],datalist[i][3]])
            i1p2.append([datalist[i][0],datalist[i][4],datalist[i][5],datalist[i][6]])
            i1p3.append([datalist[i][0],datalist[i][7],datalist[i][8],datalist[i][9]])
            i1p4.append([datalist[i][0],datalist[i][10],datalist[i][11],datalist[i][12]])
        i1p1 = i1p1[3:];i1p2 = i1p2[3:];i1p3 = i1p3[3:];i1p4 = i1p4[3:]
        # 计算均值时过滤掉无效或空的字符串
        mean_x_i1p1 = sum(safe_float_convert(point[1]) for point in i1p1 if safe_float_convert(point[1]) is not None) / \
                      len([point for point in i1p1 if safe_float_convert(point[1]) is not None]) if i1p1 else 0
        mean_x_i1p2 = sum(safe_float_convert(point[1]) for point in i1p2 if safe_float_convert(point[1]) is not None) / \
                      len([point for point in i1p2 if safe_float_convert(point[1]) is not None]) if i1p2 else 0
        mean_x_i1p3 = sum(safe_float_convert(point[1]) for point in i1p3 if safe_float_convert(point[1]) is not None) / \
                      len([point for point in i1p3 if safe_float_convert(point[1]) is not None]) if i1p3 else 0
        mean_x_i1p4 = sum(safe_float_convert(point[1]) for point in i1p4 if safe_float_convert(point[1]) is not None) / \
                      len([point for point in i1p4 if safe_float_convert(point[1]) is not None]) if i1p4 else 0

        mean_x_values = [
            ("i1p1", mean_x_i1p1),
            ("i1p2", mean_x_i1p2),
            ("i1p3", mean_x_i1p3),
            ("i1p4", mean_x_i1p4)
        ]

        # 进行排序
        sorted_mean_x_values = sorted(mean_x_values, key=lambda x: x[1], reverse=True)

        # 获取第二大均值
        second_highest_mean_x = sorted_mean_x_values[1][0]
        # 根据第二大均值的标签分配相应的点集给 new
        if second_highest_mean_x == "i1p1":
            targe = i1p1
        elif second_highest_mean_x == "i1p2":
            targe = i1p2
        elif second_highest_mean_x == "i1p3":
            targe = i1p3
        elif second_highest_mean_x == "i1p4":
            targe = i1p4
        new1 = filt(targe)
        return new1
        
    '''
    for i in [new1, new2]:
            
        np_new = np.array(i)
        x = np_new[:,1]
        y = np_new[:,2]
    
        plt.scatter(x,y)
        plt.show()
    '''
    '''
    玛曲12米
    changename = ['/home/hjh/DLC1/output/jump/MQ_jump/MQ_jump_12/T27_0644_28_0011_1DLC_resnet50_MQ_jump_12MS1Apr22shuffle1_50000_bx_filtered.csv',
                  '/home/hjh/DLC1/output/jump/MQ_jump/MQ_jump_12/T33_0477_34_7143_1DLC_resnet50_MQ_jump_12MS1Apr22shuffle1_50000_bx_filtered.csv',
                  ]
    '/home/hjh/DLC1/output/jump/MQ_jump/MQ_jump_6/T128_2830_122_0635_2DLC_resnet50_MQ_jump_6MApr16shuffle1_50000_bx_filtered.csv'
    if path in changename:
    #如果path在需要转换的名单中，则转换    
        C = Change(new1, new2)
        C.combin()
        new1 = C.new_seq1[:,:-1]
        new2 = C.new_seq2[:,:-1]
    '''
    if 'T5_026_3' in path or 'T90_075_1' in path:
        new1.extend(new2)
        new2 = 0
        return new1
    if "S38_093_35_069_3DLC" in path or 'S74_084_75_063_1' in path or 'S40_032_41_100_2' in path or 'S60_066_20_007_1' in path:
        new1 = np.array(new1)
        new2 = np.array(new2)
        new1 = new1[new1[:,2]<700]
        new2 = new2[new2[:,2]<700]
    C = Change(new1, new2)
    C.combin()
    
    try:
        new1 = C.new_seq1[:,:-1]
    except IndexError:
        new1 = np.array([])
    try:
        #print(type(C.new_seq2),C.new_seq2)
        new2 = C.new_seq2[:,:-1]
    except (IndexError,TypeError):
        new2 = np.array([])
    if individual==1:
        return new1
    else:
        try:
            return new2
        except AttributeError:
            return np.array([])
#判断何时进入上升区域
'''
————————————————————————————————————————————————————————————
用低通滤波后的数据判断起落点，输出s和t
————————————————————————————————————————————————————————————t是时间序列号
'''      
def tendst(new,wind):
    #趋势与极值判断
    '''
        new:过滤后的点图
        wind：窗格大小
        s,t是时间序列号
    '''
    #new = wherejump(point2)
    data = np.array(new)
    data = data[:,2]
    
    #低通滤波器
    b, a = signal.butter(5, 0.067, 'lowpass')   #配置滤波器
    filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
    data = filtedData
    
    new_filt = copy.deepcopy(new[:])
    for i in range(len(new)):
        new_filt[i][2] = filtedData[i]
        
        
    
    #滤波后图像
    '''
    np_new = np.array(new_filt)
    x = np_new[:,1]
    y = np_new[:,2]
    plt.scatter(x,y)
    plt.show()
    
    np_new = np.array(new)
    x = np_new[:,1]
    y = np_new[:,2]
    plt.scatter(x,y)
    plt.show
    '''
    flags = []   #建立flag集合的时候要用被过滤后的点集——new_filt
    for i in range(len(new_filt)-wind): 
        part = new_filt[i:i+wind]
        sums = 0
        sumt = 0
        for j in range(int(wind/3)):
            
            if part[j][2] == '':
                part[j][2] == 0
                    
            sums += part[j][2]
        try:
            for j in range(int(wind*2/3),wind):
                if part[j][2] == '':
                    part[j][2] == 0
                sumt += part[j][2]
            if sums < sumt:
                flags.append(1)    #后大于前标记为1 
            else:
                flags.append(2)    #前大于后标记为2 
        except:
            continue
    #print(flags)
    #最大值高度值/最低坐标值
    #plt.scatter(new_filt[:,1],new_filt[:,2])
    sort1 = sorted(new_filt, key=lambda x: x[2])
    #print(sort1)
    minp_average = (sort1[-1][0] + sort1[-2][0] + sort1[-3][0] + sort1[-4][0] + sort1[-5][0] )/5
    #print(minp_average,'QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ')
    #print(minp,flags)
    #寻找上升与下降变化的位置,找到后输出起跳开始s和落地t
    up = []
    down = []
    for i in range(len(flags)-1):
        if flags[i+1]-flags[i] == 1:
            down.append(new_filt[i][0])
        if flags[i+1]-flags[i] == -1:
            up.append(new_filt[i][0])
    fg = -1
    down = up
    #print(flags)
    #down_np =  np.array(down)
    #down_np_d = down_np - minp_average
    #index = np.where(down_np==np.min(down_np_d))[0][0]
    print('up:',down,minp_average)
    
    for i in range(len(down)):           #从右往左的跳跃此处也要进行修改**************************************************
        d2 =  abs(down[i] - minp_average)
        d1 =  abs(down[i-1] - minp_average)
        if down[i] > minp_average:
            if down[i] - down[i-1] > 50:
                t = down[i]
                s = down[i-1]
                fg = 0 
                break
#            t = down[i]
#            s = down[i-1] 
            #取极值点前后最近的拐点
            elif (d2 <= d1):
                
                t = down[i]
                fg = 2
                break
            else:
                s = down[i-1]
                break
    #if len(down) == 1 :
    #    down[0]
    #if fg == -1:
        
    '''
    #MQ 当所有拐点都在最高点左边或者右边时触发。
    if np.min(down) > minp_average:
        t = np.min(down)    #*****************夏河数据s/t互换
        fg = 2
    elif np.max(down) < minp_average:
        s = np.max(down)     #*******************夏河数据s/t互换
        fg = 1
    '''
    

    #XH
    if np.max(down) < minp_average:
        s = np.max(down)    #*****************夏河数据s/t互换
        fg = 1
    elif np.min(down) > minp_average:
        t = np.min(down)     #*******************夏河数据s/t互换
        fg = 2

    #***************************************XH/MQ需要修改
    try:
        if fg==1:
            t = minp_average*2 - s
        elif fg==2:
            s = minp_average - (t - minp_average)
    except UnboundLocalError:
        pass
        
    newf = pd.DataFrame(new,columns=['f','x','y','p'])
    try:
        s_x =newf.query('x >= 1200')['f'].iloc[-1] #t_x =newf.query('x >= 1250')['f'].iloc[0]   #6M可以用1400-750  
    except IndexError:
        print('x=1250以后该序列没有值')
        s_x = newf['f'].iloc[-1] 
    try:
        t_x = newf.query('x <= 750')['f'].iloc[0]
    except IndexError:
        t_x = newf.query('x <= 1000')['f'].iloc[0]
    try:
        if (abs(s - minp_average) < 30) or (abs(t - minp_average) <30): #如果s和t之间相隔的帧数小于50，说明识别起落点上出现了问题，所以选用在该缩放下的固定起落点（该段用固定起落点的x值获取其帧数值）#
             s = s_x
             t = t_x
             print(s,t)
    except UnboundLocalError:
             s = s_x
             t = t_x
             print(s,t)                     
#            if 1600 >= new[i][1] >= 300:
#            if 1350 >= new[i][1] >= 600:  #5M
#            if 1400>= new[i][1] >= 700:  #7.5/6米*************************************
#            if 1150 >= new[i][1] >= 800:   #10/12米*******************************    
    return s,t,minp_average
def high(new,s=0,t=0):
    #**********************根据不同缩放的模型需要有所改变*****************************
    avglist = []
    '''
    if t - s <= 30:
        s = 300  #5M 300 7.5M 500
        t = 1660  #5M 1560 7.5M 1460
    for i in range(len(new)):
        if 100 <= new[i][1] <= s :
            avglist.append(new[i][2])
        if t <= new[i][1] <= 1860 :
            avglist.append(new[i][2])
        else:
            continue
    '''
    for i in range(len(new)):   #纵向：10M 355-570  根据MQ_6M调整：
        #if (50 <= new[i][1] <= 520): # and ((405 <= new[i][2] <= 550) or (600 <= new[i][2] <= 730))
         #   avglist.append(new[i][2])
        '''MQ6M
        if (t <= new[i][0] <= t+50) and ((400 <= new[i][2] <= 800)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])     
        if (s-50 <= new[i][0] <= s) and ((400 <= new[i][2] <= 800)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])
        '''
        '''
        #XH7.5M
        
        if (t <= new[i][0] <= t+30) and ((550 <= new[i][2] <= 850)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])     
        if (s-30 <= new[i][0] <= s) and ((550 <= new[i][2] <= 850)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2]) 
        else:
            continue
        '''
        #XH10M
        
        if (t <= new[i][0] <= t+30) and ((400 <= new[i][2] <= 550)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])     
        if (s-30 <= new[i][0] <= s) and ((400 <= new[i][2] <= 550)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2]) 
        else:
            continue     
    #print(avglist)
    avg = np.mean(avglist)
    print('地面高度为：{}'.format(avg))
    return avg

        

if __name__ == '__main__':
    
    
    path = '/home/hjh/DLC1/output/jump/MQ_jump/MQ_jump_12/T9_0078_10_0467_1DLC_resnet50_MQ_jump_12MS1Apr22shuffle1_50000_bx_filtered.csv'
    
    
    new1= pretreatpoint(path,individual=1)
    new2= pretreatpoint(path,individual=2)
    for i in [new1, new2]:
            
        np_new = np.array(i)
        x = np_new[:,1]
        y = np_new[:,2]
    
        plt.scatter(x,y)
        plt.show()

    C = Change(new1,new2)
    C.combin()
    #改变坐标系，转化为更直观的轨迹坐标
    new1 = np.array(new1)
    new2 = np.array(new2)
    
    new1[:,2]=1080-new1[:,2]
    new2[:,2]=1080-new2[:,2]
    #np.set_printoptions(suppress=True)
    
    for i in [new1, new2]:
            
        np_new = np.array(i)
        x = np_new[:,1]
        y = np_new[:,2]
    
        plt.scatter(x,y)
    plt.show()
   
'''
    for i in [new1, new2]:
            
        np_new = np.array(i)
        x = np_new[:,1]
        y = np_new[:,2]
    
        plt.scatter(x,y)
        plt.show()
    
'''
