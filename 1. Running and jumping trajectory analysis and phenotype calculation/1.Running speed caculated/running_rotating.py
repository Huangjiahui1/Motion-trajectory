#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:54:45 2022
 
@author: hjh
"""
import re
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def LOF(new, n_neighbors=20):
    new = np.array(new)
    x = new[:,1]
    y = new[:,2]
    df = []
    for i in range(len(x)):
        df.append([x[i],y[i]])
    df = np.array(df)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)  
    ypre1 = clf.fit_predict(df)
    new_data = np.hstack((new, ypre1.reshape(df.shape[0], 1)))
    normal_new_data =  new_data[new_data[:, -1] == 1]  
    new_data = pd.DataFrame(normal_new_data[:,:-1])
    new_data.columns = ['croods','x','y','likelihood']
    return new_data

def plot_scatter(x, y, order):
    params = np.polyfit(x, y, order)
    funcs = np.poly1d(params)
    ypre = funcs(x)
    plt.plot(x,ypre)
    plt.scatter(x,y)
    plt.show()
    
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            
            allname = file_dir + '/' + file   #it's a difference between linux and win
            L.append(allname)
    return L
def file_name_k(file_dir):
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file[-4:] == '.MP4':
            #print(root)
                allname = root + '/'  + file   #it's a difference between linux and win
                L.append(allname)
    return L

        
#按照不同大小的缩放尺寸视频进行分类
#path = '/media/hjh/406EB92C6EB91B9A/sheepvedios/玛曲跑步/主视频/模型/csv'
path = '/media/hjh/406EB92C6EB91B9A/sheepvedios/玛曲跑步/主视频/模型/中间'

namelist_k = file_name_k(path)
k_l_MQ_T = {'178.01':[],'157.05':[],'152':[],'177.01':[],'153.03':[]}

k_l_MQ_M = {'161.05':[],'124.5':[],'118':[],'116':[],'117':[]}

k_l = k_l_MQ_M
for root in namelist_k:
    for i in k_l.keys():    
        if i in root:
            sep = i + '/'
            k_l[i].append(root.split(sep)[-1])
#for i in k_l.values():
#    print(len(i))
def k_name(k_l,name):
    #to research the k value
    for n,klist in k_l.items():
        if name in klist:
            return float(n)
'''
————————————————————————————————————————————————————————————
Sheep_individual这个类中包含了①文件读取、②点的过滤、③worldpoint的拟合填充
————————————————————————————————————————————————————————————
'''  
class Sheep_individual():
    def __init__(self):
        self.correct_x = 0
        self.correct_y = 0
        self.worldpdic = {}
        pass

    def _creatpoint(self,pose_data,worldpointflag=False):
        #输入为pandas的DataF格式
        #类外批量添加变量
        pointlist = {}
        
        for i in range(1,len(pose_data.columns),3):
            x = [pose_data.columns[0],pose_data.columns[i],pose_data.columns[i]+'.1',pose_data.columns[i]+'.2']
            pointlist[pose_data.columns[i]] = pose_data[x][1:] #删除重复的第一行
            pointlist[pose_data.columns[i]].columns = ['croods','x','y','likelihood'] #所有变量改标题
            pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].astype(float) #全体变量改浮点数
            if worldpointflag:
                pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].query('likelihood>0.8')  #根据likelihood >0.8过滤
            else:
                pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].query('likelihood>0.6')  #worldpoint根据likelihood >0.6过滤
            pointlist[pose_data.columns[i]] =  pointlist[pose_data.columns[i]].reset_index(drop=True)

            #进行LOF过滤
            if worldpointflag:
                try:
                    pointlist[pose_data.columns[i]] = LOF(pointlist[pose_data.columns[i]],25)
                    pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].iloc[10:-30]
                except ValueError:
                    print('过滤时个别点存在数据量不够的情况')
                    continue
            if len(pointlist[pose_data.columns[i]]) < 30 :#如果点少于30个就进入下一个循环
                continue
            if worldpointflag:
                try:
                    #用于给worldpoint填充值
                    pointlist[pose_data.columns[i]] = self.worldpoint_pre(pointlist[pose_data.columns[i]], 10)#设置order阶数
                    pointlist[pose_data.columns[i]].columns = ['croods', 'x', 'y']
                    print('worldpoint填充完毕')
                    
                    x_d,y_d = self.ED(pointlist[pose_data.columns[i]])
                    pointlist[pose_data.columns[i]]['x_d'] = x_d; pointlist[pose_data.columns[i]]['y_d'] = y_d #将worldpoint的差值加入DataFrame中
                    
                except KeyError:
                    print('KeyError,数据量不够')
            setattr(self, pose_data.columns[i],pointlist[pose_data.columns[i]]) #用于给类添加属性的setattr方法
        return pointlist
    
    
    def pretreatpoint(self,pose_path,worldpoint_path):   
        '''
        individual:1
        bodypart bodypart2 mid bodypart3 objectA b6	b7 b8 lhoof	rhoof hind_lhoof hind_rhoof
        此处用于输入数据的预处理，并且每组都有pose预测数据和world世界坐标数据
        '''

        pose_data = pd.read_csv(pose_path,keep_default_na=False,header=1)
        worldpoint_data = pd.read_csv(worldpoint_path,keep_default_na=False,header=1)

        #根据pose_data中所有识别的点自动创建变量，每个变量对应的数据以 ['crood','x','y','likelihood']格式
        self.posepdic = self._creatpoint(pose_data)  #生成储存点轨迹的字典文件
        self.worldpdic = self._creatpoint(worldpoint_data, worldpointflag=True)
        #del self.worldpdic['bodypart1']; del self.worldpdic['bodypart2']; del self.worldpdic['bodypart3']; del self.worldpdic['objectA']#删除不需要的坐标数据
        
    def _funcs(self, x, y, order=6, p=False):
        #用于提取选定的两列值，按照order作为多项式拟合阶数拟合数据
        if len(x) == 0:
            print('点集为空，无法画图')
            return False
        order = 20
        params = np.polyfit(x, y, order)
        funcs = np.poly1d(params)
        ypre = funcs(x)
        #画图
        if p:
                
            plt.plot(x,ypre)
            plt.scatter(x,y)
            plt.show()
        return funcs
    
    def worldpoint_pre(self, pointname, order):
        #用于预测和填充世界坐标中的缺失值，主要是用多项式拟合的方法
        
        croods = pointname['croods']; x = pointname['x'];y = pointname['y'] #赋值
        funcs_x = self._funcs(croods, x, order=order,p=False) #数据拟合
        funcs_y = self._funcs(croods, y, order=order,p=False) #plot
        plt.show()
        
        croods_fit = [x for x in range(int(croods.iloc[0]), int(croods.iloc[-1]))]
        #print(funcs_x)
        xpre = funcs_x(croods_fit)
        ypre = funcs_y(croods_fit)
        new_worldpoint = pd.DataFrame([croods_fit,list(xpre),list(ypre)]).T
        return new_worldpoint

    def ED(self, w_point):
        '''
        距离计算,错位做差,累积加和作为每一帧的矫正值
        '''
        #用shift将所有数据的index往后位移，然后想减得到前后的间隔
        x_d = w_point['x'] - w_point['x'].shift(1)
        y_d = w_point['y'] - w_point['y'].shift(1)
        #w_point['x_d'] = x_d
        #w_point['y_d'] = y_d
        return x_d, y_d
        #return ED
    
    def create_crood_index(self, crood_wp, posepoint, worldpoint_s, worldpoint_t):
        #try:
            #用wp_s_index、wp_t_index、pp_index分别提取出该帧号处的点的索引，然后用_index信息作为统一时间信息

        wp_s_index = worldpoint_s[worldpoint_s['croods']==crood_wp].index[0]
        if len(worldpoint_t[worldpoint_t['croods']==crood_wp]) == 0:
            wp_t_index = 0
        else:
            wp_t_index = worldpoint_t[worldpoint_t['croods']==crood_wp].index[0]
        #try:
        wp_s_x = worldpoint_s['x'].iloc[wp_s_index]  #以帧号作为索引,wp(worldpoint),s(start),x(x坐标)
        self.wp_s_y = worldpoint_s['y'].iloc[wp_s_index] 
        wp_t_x = worldpoint_t['x'].iloc[wp_t_index]
        self.wp_t_y = worldpoint_t['y'].iloc[wp_t_index] 
        pp_index = posepoint[posepoint['croods']==crood_wp].index[0] #pp(posepoint) x(x坐标)
        pp_x = posepoint['x'].iloc[pp_index]
            #print(wp_s_x,wp_t_x,pp_x)
        return pp_x,wp_s_x,wp_t_x,pp_index,wp_s_index,wp_t_index

    def posepoint_correct(self, posepoint, a=0, b=0):
        #b = len(posepoint)
        #先判断动物运动到了哪个点附近,从动物过点point3开始计算，
        #其中_s是开始点，_t是结束点，例如worldpoint3-4作为输入。
        self.correct_x = 0
        self.correct_y = 0
        start = 3000
        log = []
        t = 0
        position_flag = []
        posepoint['d'] = 0
        now_d = 0
        wp_lenth = []
        d_st = 0
        for i in range(a, b):  
            #print('i===',i)
            crood0 = int(self.worldpoint3['croods'].iloc[0])

            crood_wp = int(posepoint['croods'].iloc[i]) #进入循环后，记录当前worldpoint的帧号
            
            flag_s = 1
            for d in range(1,5): #每个循环就是进入了每一段用不同的点矫正的轨迹中
                if d < now_d:
                    continue
            
                worldpoint_s = self.__dict__['worldpoint{}'.format(d)]
                worldpoint_t = self.__dict__['worldpoint{}'.format(d+1)]
                #print('假设移动到了点{}到点{}之间，i={},crood_wp={}'.format(d,d+1,i,crood_wp))
                if len(worldpoint_s[worldpoint_s['croods']==crood_wp]) == 1 : #如果进入循环的时候crood不在worldpoint_t范围内则跳过
                    pp_x, wp_s_x, wp_t_x, pp_index, wp_s_index, wp_t_index = self.create_crood_index(crood_wp, posepoint, worldpoint_s, worldpoint_t)
                    d_st =  wp_s_x - wp_t_x  
                    
                    print('crood在范围内',pp_x, wp_s_x, wp_t_x, pp_index, wp_s_index, wp_t_index)
                elif (d_st != 0) and (len(worldpoint_t[worldpoint_t['croods']==crood_wp]) == 1):
                    pp_index = posepoint[posepoint['croods']==crood_wp].index[0]
                    wp_t_index = worldpoint_t[worldpoint_t['croods']==crood_wp].index[0]
                    pp_x = posepoint['x'].iloc[pp_index]
                    wp_t_x = worldpoint_t['x'].iloc[wp_t_index]
                    
                    if wp_t_x < pp_x < wp_s_x: #用x轴的上下限来限定posepoint       
                        if crood_wp < start:
                            start = crood_wp
                        self.correct_x = self.correct_x + worldpoint_t['x_d'].iloc[wp_t_index]
                        posepoint['x'].iloc[pp_index] = posepoint['x'].iloc[pp_index] - self.correct_x #修正posepoint
                        self.correct_y = self.correct_y + worldpoint_t['y_d'].iloc[wp_t_index]
                        posepoint['y'].iloc[pp_index] = posepoint['y'].iloc[pp_index] - self.correct_y
                        #print('___XXX wp_t_x={},pp_x={},wp_s_x={}'.format(wp_t_x,pp_x,wp_s_x))
                        #print('移动到了点{}到点{}之间，i={},crood_wp={}'.format(d,d+1,i,crood_wp))
                        #print('correct_x={},correct_y={},dx = {}'.format(self.correct_x,self.correct_y, worldpoint_s['y_d'].iloc[wp_s_index]))
                        
                        x_k = (d_st-(pp_x-wp_t_x))/d_st
                        #夏河的围栏桩距离是2.83m 玛曲是3.40m,需要手动更改 ****************************
                        posepoint['d'].iloc[pp_index] = x_k*2.834
                        #print(x_k*3.40)
                        t = crood_wp
                        now_d = d
                        break
                    continue
                   
                else:
                    #print('crood不在crood{}范围内'.format(crood_wp))
                    continue
                
                if wp_t_x < pp_x < wp_s_x: #用x轴的上下限来限定posepoint
                    
                    if crood_wp < start:
                        start = crood_wp
                    x_k = (wp_s_x-pp_x)/(wp_s_x-wp_t_x)    
                    pp_y = posepoint['y'].iloc[pp_index]
                    wp_s_y = worldpoint_s['y'].iloc[wp_s_index] 
                    wp_t_y = worldpoint_t['y'].iloc[wp_t_index] 
                    y_k = (wp_s_y-pp_y)/(wp_s_y-wp_t_y)
                    #此处想用转出屏幕外时候的s、t点组合中，以t点代替s点,此为XH_T_swing特殊情况
                    self.correct_x = self.correct_x + (worldpoint_s['x_d'].iloc[wp_s_index])*(1-x_k) + (worldpoint_t['x_d'].iloc[wp_t_index])*x_k #根据两个点,累积计算correct_x相对位移
                    #self.correct_x = self.correct_x + (worldpoint_s['x_d'].iloc[wp_s_index]) #根据两个点,累积计算correct_x相对位移
                    posepoint['x'].iloc[pp_index] = posepoint['x'].iloc[pp_index] - self.correct_x #修正posepoint
                    
                    #self.correct_y = self.correct_y + (worldpoint_s['y_d'].iloc[wp_s_index])
                    self.correct_y = self.correct_y + (worldpoint_s['y_d'].iloc[wp_s_index])*(1-y_k) + (worldpoint_t['y_d'].iloc[wp_t_index])*y_k
                    
                    posepoint['y'].iloc[pp_index] = posepoint['y'].iloc[pp_index] - self.correct_y
                    #print('XXX wp_t_x={},pp_x={},wp_s_x={}'.format(wp_t_x,pp_x,wp_s_x))
                    #print('移动到了点{}到点{}之间，i={},crood_wp={}'.format(d,d+1,i,crood_wp))
                    #print('correct_x={},correct_y={},dx = {}'.format(self.correct_x,self.correct_y, worldpoint_s['y_d'].iloc[wp_s_index]))
                    count_d = np.array([wp_t_x,self.wp_t_y]) - np.array([wp_s_x,self.wp_s_y])
                    #夏河的围栏桩距离是2.83m 玛曲是3.40m,需要手动更改 ****************************
                    posepoint['d'].iloc[pp_index] = x_k*2.834
                    #print(x_k*3.40)
                    t = crood_wp
                    now_d = d
                    #position_flag.append([crood_wp,d,k])
                    break  #这里的break很重要，因为在上一次的纠正中被改变的x值可能导致下个小循环中满足了另一个if的阈值，导致重复循环

        print('start',start) #这里标记了矫正是从第多少帧开始进行的
        return start,t

'''
————————————————————————————————————————————————————————————
接下来需要实现步态周期的计算和分割
————————————————————————————————————————————————————————————
方案:40frame滑窗，找到范围内的最大值并标记，然后将标记相近的比大小合并。
'''  
class Cycle_count_peak():
    def __init__(self):
        pass
    
    def  peak_n(self,posepoint, start, n_crood=40):
        #从周围n个值中找到最大值并且加入某个DF中
        max_df = pd.DataFrame([])
        posepoint_max = posepoint.query('croods>@start')
        
        for i in range(len(posepoint_max)):
    
            crood = posepoint_max[i:i+1]['croods'].iloc[0]
            if crood-n_crood < posepoint_max[:1]['croods'].iloc[0]:
                start = 0
            else:
                start = crood-n_crood
            if crood+n_crood > posepoint_max[-1:]['croods'].iloc[0]:
                ter = posepoint_max[-1:]['croods'].iloc[0]
            else:
                ter = crood+n_crood
            #print(posepoint_max.query('@start < croods < @ter').sort_values(by=['y']))
            if len(posepoint_max.query('@start < croods < @ter')) == 0:
                continue
            max_series = posepoint_max.query('@start < croods < @ter').sort_values(by=['y']).iloc[-1]
            max_df = max_df.append(max_series)
        max_df.drop_duplicates(subset=['croods'],keep='first',inplace=True) #去重复
        max_df = max_df.sort_values(by = ['x'])
        
        maxpoint = pd.DataFrame([])
        #对筛选出的局部最高点进行合并，将距离在上下100范围内的合并起来
        max_df.sort_values(by=['croods'], inplace=True)
        #max_df.drop(max_df[max_df['x'] - max_df['x'].shift(1)>0].index)
        max_df.sort_values(by=['x'], inplace=True)
        while True:
            x = max_df['x'].iloc[0]
            targe = max_df.query('@x-100 < x < @x+120') #
            series = targe.sort_values(by=['y']).iloc[-1]
            maxpoint = maxpoint.append(series)
            max_df = max_df[len(targe):] #删掉已经被合并的数据    
            if len(max_df) == 0:
                break
        
        maxpoint.sort_values(by=['croods'], inplace=True)
        x_d = maxpoint['x'] - maxpoint['x'].shift(1)
        #print(maxpoint[x_d>0].index)
        maxpoint = maxpoint.drop(maxpoint[x_d>0].index)
        #合并较小的cycle
        return maxpoint
    
    def maxpoint_filt(self,point,maxpoint):
        maxpoint = maxpoint.sort_values(by=['croods'])
        crood_d = (maxpoint['croods'] - maxpoint['croods'].shift(1))/30
        maxpoint['cycle'] = crood_d 
        #print(maxpoint[['croods','cycle']])
        #过滤较小的cycle，靠近的选更高的
        for i in range(1,len(maxpoint)-1):
            if maxpoint['cycle'].iloc[i] < 1.2: #如果一个周期小于1.2视频时长，则将其向前后合并
                    
                if abs(maxpoint['y'].iloc[i] > maxpoint['y'].iloc[i-1]):  #如果离上一个周期更近，就合并进去，否则合并进下一个周期
                    maxpoint['cycle'].iloc[i] = maxpoint['cycle'].iloc[i] + maxpoint['cycle'].iloc[i-1]
                    maxpoint['cycle'].iloc[i-1] = 0
                else:   
                    maxpoint['cycle'].iloc[i-1] = maxpoint['cycle'].iloc[i] + maxpoint['cycle'].iloc[i-1]
                    maxpoint['cycle'].iloc[i] = 0
            #排除假阳性的点,以局部最高点在point集合中周围支持的点个数为标准
        maxpoint = maxpoint.query('cycle!=0')
        for i in range(len(maxpoint)):
            p = maxpoint['x'].iloc[i]
            if len(point.query('@p-100<x<@p+100')) < 15:
                maxpoint['cycle'].iloc[i] = 0
        maxpoint = maxpoint.query('cycle!=0')
        plt.scatter(point['x'],point['y'])
        plt.scatter(maxpoint['x'],maxpoint['y'])
        plt.show()
        return maxpoint
    
class Cycle_count_touchtime():
    def _cycle_count(self,point,seppoint):        
        cyclelist = []
        seppoint = pd.DataFrame(seppoint)
        for i in range(len(seppoint)):
            if i == 0:
                t = seppoint.iloc[i+1][0]
                s = seppoint.iloc[i][0]
                cyclelist.append(point.query('@s<=croods<@t'))
            elif i == len(seppoint) - 1:
                s = seppoint.iloc[i][0]
                cyclelist.append(point.query('croods>=@s'))
            else:
                s = seppoint.iloc[i][0]
                t = seppoint.iloc[i+1][0]
                cyclelist.append(point.query('@s<=croods<@t'))
        return cyclelist

    def _get_cycle(self,point): #Used to divide the period
        point['d'] = point['x'] - point['x'].shift(1)
        stable_list = []
        range_list = []
        for i in range(0,len(point)):
            #print(cycle.croods.iloc[i],abs(cycle[i:i+5].d.sum()))
            if abs(point[i:i+5].d.sum()) < 15:
                stable_list.append(point.croods[i:i+3].iloc[-1])  #i -> i+5  ############!!!!!!!!!!!!!!change it
        stable_df = pd.DataFrame(stable_list)
        stable_df['d'] = stable_df - stable_df.shift(1)
        d_point = stable_df.query('d>10')
        a = 0; b = 0
        for i in range(len(d_point)):
            if i == 0:
                b = d_point.index[0]-1
                range_list.append([stable_df.iloc[0][0],stable_df.iloc[b][0]])
            else:
                a = d_point.index[i-1]
                b = d_point.index[i]-1
                range_list.append([stable_df.iloc[a][0],stable_df.iloc[b][0]])
        a = d_point.index[-1]
        range_list.append([stable_df.iloc[a][0],stable_df.iloc[-1][0]])
        return range_list

    '''
    ————————————————————————————————————————————————————————————
    用左前肢触地的一瞬间作为步态周期的开始，以连续的下一次左前肢触地为结束一个步态周期
    ————————————————————————————————————————————————————————————
    '''  
    def stable_start(self, point):
        range_list = self._get_cycle(point)
        self.stable_range_list = []
        cycle_list = []
        for range1 in range_list: #用peak峰值方法算出的结果结合验证,得到一组较为准确的触地时期。用触地时期计算步态周期
        #用取并集的方法，如果一个range没有被maxpoint选中，并且其长度小于一定值（15），则舍弃掉
            a,b = range1
            if abs(a-b) >= 8: #pass
                #print([a,b])
                self.stable_range_list.append([a,b])
                continue
            for i in range(len(maxpoint)):
                if maxpoint.croods.iloc[i] in [i for i in range(int(a),int(b))]:
                    self.stable_range_list.append([a,b])
                    
        stable_start_array = np.array(self.stable_range_list)[:,0]
        cycle_range_list = []
        for i in range(len(stable_start_array)):
            if i == len(stable_start_array)-1:
                cycle_range_list.append([stable_start_array[i],self.stable_range_list[-1][-1]])
            else:
                cycle_range_list.append([stable_start_array[i],stable_start_array[i+1]])
            print('cycle_range_list:',cycle_range_list)
        self.cycle_range_list = cycle_range_list
        return stable_start_array
    
     #用cycle_count分割步态周期##########
    '''
    ————————————————————————————————————————————————————————————
    用左前肢触地步态周期作为划分标准，用cycle_count分割步态周期
    ————————————————————————————————————————————————————————————
    '''  
    def create_dict(self,stable_start_array):
        self.limbs_list = ['lhoof','rhoof','hind_lhoof','hind_rhoof']
        self.cycle_dict = {}
        for name in self.limbs_list:
            point_n = p_class.__dict__[name]
            self.cyclelist = self._cycle_count(point_n,stable_start_array)
            self.cycle_dict[name] = self.cyclelist
        #plot
        q = 221
        for cyclelist in self.cycle_dict.values():
            plt.subplot(q)
            q += 1
            for cycle in cyclelist[:]:
                plt.scatter(cycle['croods'],cycle['x'])
        plt.show()
        return self.cycle_dict
    #计算触地时间以lhoof为标准
    #找到时序数据的平稳位置，以计算触地时间
    def _stable_time(self, cycle): #use to count the others cycle #######################
        cycle['d'] = cycle['x'] - cycle['x'].shift(1)
        stable = []
        d10 = cycle.query('d>10')
        x10 = cycle.query('d<-10')

        if len(d10>0): #清除来回振动,找到一个d>10的最大差值点，进一步寻找它附近的反向振动
            for _max in d10.sort_values(by=['d']).index:
                try:
                    _min = cycle.loc[_max-10:_max+10].query('d<10').sort_values(by=['d']).index[0]
                except:
                    continue
                if abs(cycle.loc[_min].d) > abs(cycle.loc[_max].d):
                    cycle.loc[_min].d = cycle.loc[_min].d + cycle.loc[_max].d
                    cycle.loc[_max].d = 0
                    #print('_min:{},_max:{}抵消'.format(_min,_max))
                elif abs(cycle.loc[_min].d) < abs(cycle.loc[_max].d):
                    cycle.loc[_max].d = cycle.loc[_min].d + cycle.loc[_max].d
                    cycle.loc[_min].d = 0
                    #print('_min:{},_max:{}抵消'.format(_min,_max))

        for i in range(0,len(cycle)):            
            #一阶差分 往前5格滑窗+向后5格滑窗        limbs_all_dict[str(i)].to_excel('test_{}.xlsx'.format(i))
            if (abs(cycle[i:i+5].d.sum()) < 10): #and (i<5)
                stable.append(cycle.croods.iloc[i])
            #elif (abs(cycle[i:i+5].d.sum()) < 10) or (abs(cycle[i-4:i+1].d.sum()) < 10): #and (len(cycle[i:i+5])>2): #清除集合中只有一个点的情况（末尾点）
            #    stable.append(cycle.croods.iloc[i])
        return stable
    def count_touchdown_time(self):
        limbs_cycle_dict = {'cycle_name':[i for i  in range(len(self.cyclelist))],'lhoof_1':0,'lhoof_2':0,'lhoof_time':0,'rhoof_1':0,'rhoof_2':0,'rhoof_3':0,\
                            'hind_lhoof_1':0,'hind_lhoof_2':0,'hind_lhoof_3':0,'hind_rhoof_1':0,'hind_rhoof_2':0,'hind_rhoof_3':0}
        limbs_cycle_df = pd.DataFrame(limbs_cycle_dict) 

        for name,cyclelist in self.cycle_dict.items():
            if name == 'lhoof':
                for j in range(len(self.stable_range_list)): 
                    stable = self.stable_range_list  #######################
                    limbs_cycle_df[name+'_1'][j] = str([stable[j][0],stable[j][1]])
                    #print(str([stable[j][0],stable[j][1]]))
                continue
            for i in range(len(cyclelist)):
                
                stable = self._stable_time(cyclelist[i]) #######################
                print(stable)
                if len(stable) < 2 :
                    print(stable)
                    continue
                
                stable_df = pd.DataFrame(stable)
                stable_df['d'] = stable_df[0] - stable_df[0].shift(1)
                d_point = stable_df.query('d>5')
                t = 3
                if (len(d_point)>1) and (d_point.index[-1] == stable_df.index[-1]): #将末尾单个的d_point点删除
                    d_point = d_point[:-1]
                if len(d_point) == 0:
                    if abs(stable[0] - stable[-1]) > t:
                        limbs_cycle_df[name+'_1'][i] = str([stable[0],stable[-1]])
                if len(d_point) ==1:
                    ###########range小于t的范围被认为是不完整的触地或者错误识别的触地
                    if (abs(stable[0] - stable_df.iloc[d_point.index[0]-1].iloc[0]) < abs(stable_df.iloc[d_point.index[0]].iloc[0]-stable[-1])) and (abs(stable_df.iloc[d_point.index[0]].iloc[0]-stable[-1]) > t):
                        limbs_cycle_df[name+'_1'][i] = str([stable_df.iloc[d_point.index[0]].iloc[0],stable[-1]])
                    if (abs(stable[0] - stable_df.iloc[d_point.index[0]-1].iloc[0]) > abs(stable_df.iloc[d_point.index[0]].iloc[0]-stable[-1])) and (abs(stable[0] - stable_df.iloc[d_point.index[0]-1].iloc[0]) > t):
                        limbs_cycle_df[name+'_1'][i] = str([stable[0],stable_df.iloc[d_point.index[0]-1].iloc[0]])

                if len(d_point) ==2:
                    #############
                    range1 = abs(d_point.index[0] - stable_df.index[0])
                    range2 = abs(d_point.index[0] - d_point.index[-1])
                    range3 = abs(d_point.index[-1] - stable_df.index[-1])
                    if range1 == max(range1,range2,range3):
                        limbs_cycle_df[name+'_1'][i] = str([stable[0],stable_df.iloc[d_point.index[0]-1].iloc[0]])
                    elif range2 == max(range1,range2,range3):
                        limbs_cycle_df[name+'_1'][i] = str([stable_df.iloc[d_point.index[0]].iloc[0],stable_df.iloc[d_point.index[1]-1].iloc[0]])
                    else:
                        limbs_cycle_df[name+'_1'][i] = str([stable_df.iloc[d_point.index[1]].iloc[0],stable[-1]])

        return limbs_cycle_df
    
def Corrected_rotation(sheep):
    list_p = [sheep.lhoof,sheep.rhoof,sheep.hind_lhoof,sheep.hind_rhoof]
    for point in list_p:
        p1 = point.copy()
        start = sheep.posepoint_correct(point,0 ,len(point))
        point = point.query('croods>@start')
        p1 = p1.query('croods>@start')
        plt.scatter(point['croods'],point['x'])
        plt.scatter(p1['croods'],p1['x'])    
        #plt.title('%s' % p) 
        plt.show()

def count_V(point,k):
    # k is a scale of the picture
    dt = (point.iloc[-5:]['croods'].mean() - point.iloc[:5]['croods'].mean())/180
    dx = (point.iloc[:5]['x'].mean() - point.iloc[-5:]['x'].mean())/k
    v = dx/dt
    cycle_time = (point.iloc[-1]['croods'] - point.iloc[0]['croods'])/180
    return v,cycle_time
def get_list(str1):
    t=re.findall(r'\[(.+?)\]',str1)
    t1=t[0].split(',')
    b=[float(i) for i in t1]
    return b

def creat_idic(pose_path_fir, worldpoint_path_fir):
    #用于创建类的个体数据实例化
    idic = {}
    for i in range(len(pose_path_fir)): #len(pose_path_fir)
        try:
            idic['{}'.format(i)] = Sheep_individual()
            idic['{}'.format(i)].pretreatpoint(pose_path_fir[i],worldpoint_path_fir[i])
            point = idic['{}'.format(i)].lhoof
            p1 = point.copy()
            print(pose_path_fir[i].split('/')[-1].split('DLC')[0],'轨迹创建完毕,i=',i)
            e=0
        
            start,t = idic['{}'.format(i)].posepoint_correct(idic['{}'.format(i)].lhoof,0 ,len(point))

            idic['{}'.format(i)].lhoof = idic['{}'.format(i)].lhoof.query('@t>croods>@start')
            
            point = point.query('croods>@start')
            p1 = p1.query('croods>@start')
            plt.scatter(point['croods'],point['x'])
            plt.scatter(p1['croods'],p1['x'])    
            name = pose_path_fir[i].split('/')[-1].split('DLC')[0]
            plt.title(name)
            
            plt.savefig("/home/hjh/DLC1/output/run/MQmid/run_csv_fig/{}.jpg".format(name))
            plt.show()
            idic['{}'.format(i)].lhoof_old = p1.query('@t>croods>@start')
        except (AttributeError,KeyError):
            e=e+1
            pass
    print(e)
    return idic

if __name__ == '__main__':
    pose_path = '/home/hjh/DLC1/output/run/XH_jump_T/20220522_Tswing/20220522_run_posepre'
    worldpoint_path = '/home/hjh/DLC1/output/run/XH_jump_T/20220522_Tswing/20220522_run_worldpoint'
    #pose_path = '/home/hjh/DLC1/output/run/MQmid/117_posepoint'
    #worldpoint_path = '/home/hjh/DLC1/output/run/MQmid/117_worldpoint'
    
    pose_path_fir = file_name(pose_path)
    pose_path_fir.sort()
    worldpoint_path_fir = file_name(worldpoint_path)
    worldpoint_path_fir.sort()

#创建类的实例化个体
    idic = creat_idic(pose_path_fir, worldpoint_path_fir)
    
#前处理完成，开始后续分析
    limbs_all_dict = {}
    P = Cycle_count_peak()
    T = Cycle_count_touchtime()
    
    df_alldata = pd.DataFrame([])
    error = 0
    for i in range(len(idic)):#len(idic)
        print(('{}%'.format(100*(i/len(idic)))))
        try:
            p_class = idic[str(i)]
            id1 = str(i)
            #刷新class
            #p_class.pretreatpoint(pose_path_fir[17])
            point = p_class.lhoof
            T = Cycle_count_touchtime()
            #peak_n to count maxpoint
            maxpoint = P.peak_n(point,1)
            #maxpoint_filt
            maxpoint = P.maxpoint_filt(point,maxpoint)
            #cycle_count
            '''
            ————————————————————————————————————————————————————————————
            用左前肢触地的一瞬间作为步态周期的开始，以连续的下一次左前肢触地为结束一个步态周期
            ————————————————————————————————————————————————————————————
            '''  
            
            #cyclelist = T.cycle_count(point,maxpoint)
            stable_start_array = T.stable_start(point)
            if len(stable_start_array) == 0: #############################################
                print('stable_start=',0)
                continue
            cycle_dict = T.create_dict(stable_start_array)  #self.cycle_dict
            
            limbs_cycle_df = T.count_touchdown_time()
            limbs_all_dict[str(i)] = limbs_cycle_df
            limbs_cycle_df = limbs_cycle_df.loc[:, (limbs_cycle_df != 0).any(axis=0)]  #删除全部为0的那一列
            '''
            ————————————————————————————————————————————————————————————
            进行步态分类
            ————————————————————————————————————————————————————————————
            '''  
            gait_df = pd.DataFrame({'cycle_name':[i for i  in range(len(limbs_cycle_df))],'Landing order':0,'V':0, 'cycle_time':0})
            raw = 0 #用以计数遍历到达了第几行
            for cycle in limbs_cycle_df.itertuples(): #完成字符串到列表的转换
                a_list = []
                for i in cycle[2:]:
                    if i == 0:
                        a_list.append(0)
                        continue
                    a_list.append(get_list(i)[0])        
                a_df = pd.DataFrame(a_list)
                a_df = a_df.sort_values(by=[0])[a_df[0]>0] #对周期的开始节点进行排序
                r = ''
                for order in a_df.T:
                    r = r + str(order+1)
                r = r.ljust(4,'X') #将为0或者计算错误的值补齐
                #print(r)
                gait_df['Landing order'].iloc[raw] = r
                raw += 1

            '''
            ————————————————————————————————————————————————————————————
            以步态周期为单位计算奔跑速度
            ————————————————————————————————————————————————————————————
            '''  
            name = pose_path_fir[int(id1)].split('/')[-1]
            
            point = p_class.bodypart2
            seppoint = stable_start_array
            #######
            bodypart2 = idic[id1].bodypart2
            p1 = bodypart2.copy()
            #print(pose_path_fir[i].split('/')[-1].split('DLC')[0],'轨迹创建完毕,i=',i)
        
            start,t = idic[id1].posepoint_correct(idic[id1].bodypart2,0 ,len(bodypart2))
            point = bodypart2.query('croods>@start')
            p1 = p1.query('croods>@start')
            plt.scatter(point['croods'],point['x'])
            plt.scatter(p1['croods'],p1['x'])    
            idic[id1].bodypart2 = idic[id1].bodypart2.query('@t>croods>@start')
            print('s={},t={}'.format(start,t))
            plt.show()
            
            #get the V
            point_cycle_list = T._cycle_count(point,seppoint)
            if len(point_cycle_list[-1]) == 0:
                point_cycle_list = point_cycle_list[:-1]
            point_cycle_list[-1] = point_cycle_list[-1].query('d!=0')
            V_list = []
            for cyc in point_cycle_list:
                if len(cyc) ==0 :
                    continue
                cyc['d_d'] = cyc.d - cyc.d.shift(1)
                cyc = cyc.query('d_d>-1')
                s = cyc.d_d.sum()
                t = (cyc.croods[-1:].iloc[0] - cyc.croods[:1].iloc[0])/180
                V = s/t
                print(V)
                V_list.append([V,t])
                
            for i in range(len(V_list)):
                gait_df['V'].iloc[i] = V_list[i][0]
                gait_df['cycle_time'].iloc[i] = V_list[i][1]
            merge_df = pd.concat([limbs_cycle_df,gait_df.iloc[:,1:]],axis=1)        
            merge_df_head = pd.DataFrame({'cycle_name':[name.split('DLC')[0]]}).append(merge_df)
            df_alldata = df_alldata.append(merge_df_head)
        except:
            print('error')
            error += 1
        
    #excel的格式化输出
    excel = pd.ExcelWriter('XH_Tswing_0623.xlsx')
    df_alldata.to_excel(excel, 'aa', index=None)
    sheet = excel.sheets['aa']
    sheet.set_column(1, 10, 12)
    sheet.set_column(0, 0, 12)
    #sheet.set_row(1,None, )
    excel.close()
    df_alldata = df_alldata.reset_index(drop=True)
    
    df_alldata = df_alldata.where(df_alldata.notnull(), '-')
    df_alldata_str = df_alldata.astype("str")
    df_alldata_str_cycle01_V1to10 = df_alldata_str.query('"0.5">cycle_time>"0.1" or V=="-"').query('(V>"1") or (V =="-")')
    
    df_alldata_str_cycle01_V1to10 = df_alldata_str_cycle01_V1to10.query('((lhoof_1!="-") and (V!="-")) or (lhoof_1=="-")')

    df_alldata_str_cycle01_V1to10.to_excel('XH_Tswing_0623_filt.xlsx')
    
    df_alldata_str_filt = pd.read_excel('XH_Tswing_0623_filt_hum.xlsx')
    
    #df_alldata_str_filt = pd.read_excel('20220523_xhfixed_T_filt.xlsx')
    #读取过来统一进行过滤处理
    df_alldata_str_filt = pd.read_excel('MQ_T_all_V_0610_filt.xlsx')
    df_alldata_str_filt = df_alldata_str_filt.reset_index().drop('index',axis=1) #重置行索引,提取个体分割的索引值，提取多个周期中最大的三个速度值
    index_list = df_alldata_str_filt.query('cycle_time == "-"').index
    for i in range(len(index_list)):
        if i == len(index_list)-1:
            s = index_list[i]
            t = -1
        else:    
            s = index_list[i]
            t = index_list[i+1]
        series = df_alldata_str_filt[s:t].sort_values(by=['V'])[-3:].query('V!="-"').V.astype('float')
        if len(series) == 0:
            continue
        while 1:
            if series.iloc[-1] - series.iloc[0] > 0.5: ##################################
                series = series[1:]
                continue
            else:
                break
        V_max = series.mean()  #提取对应区段，找到速度最大的三个值，求他们的平均值
        df_alldata_str_filt.iloc[s,6] = V_max
        
    df_alldata_str_filt.query('lhoof_1=="-"')[['cycle_name','V']].to_excel('XH_Tswing_name_V_0623.xlsx')
    V_max_df = df_alldata_str_filt.query('cycle_time=="-"')[['cycle_name','V']]
    V_max_df = V_max_df.query('V!=0')
    V_max_df.to_excel('XH_Tswing_V_0623.xlsx')
