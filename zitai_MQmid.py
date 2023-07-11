#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:10:50 2021

@author: hjh
"""
import copy
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import math
#定义一个类用来存放个体的运动数据,以个体为单位`
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
    toal_data = np.hstack((df, ypre1.reshape(df.shape[0], 1)))
    new_data = np.hstack((new, ypre1.reshape(df.shape[0], 1)))
    normal_new_data =  new_data[new_data[:, -1] == 1]  
    new_data = pd.DataFrame(normal_new_data[:,:-1])
    new_data.columns = ['croods','x','y','likelihood']
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
    new_data.columns = ['croods','x','y','likelihood']
    return pd.DataFrame(new_data)
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
'''
————————————————————————————————————————————————————————————
Sheep_individual这个类中包含了①文件读取、②点的过滤、③worldpoint的拟合填充，以及④通过worldpoint对动物体标记点的矫正
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
             #新模型的数据没有这个问题
            if worldpointflag:
            #对世界坐标进行操作的时候跳过前面的body部分
                if i <= 12:
                    continue
            
            x = [pose_data.columns[0],pose_data.columns[i],pose_data.columns[i]+'.1',pose_data.columns[i]+'.2']
            pointlist[pose_data.columns[i]] = pose_data[x][1:] #删除重复的第一行
            pointlist[pose_data.columns[i]].columns = ['croods','x','y','likelihood'] #所有变量改标题
            pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].astype(float) #全体变量改浮点数
            pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].query('likelihood>0.9')  #根据likelihood >0.8过滤
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
                    pointlist[pose_data.columns[i]] = self.worldpoint_pre(pointlist[pose_data.columns[i]], 8)#设置order阶数
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
        #print(croods,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        funcs_x = self._funcs(croods, x, order=order,p=True) #数据拟合
        funcs_y = self._funcs(croods, y, order=order,p=True) #plotplotplotplotplotplotplotplotplotplotplotplotplotplotplotplot
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
        '''
        except IndexError:
            wp_t_index =  0 #该点还没进入范围，故用最小值代替
            return 
        '''
        #try:
        wp_s_x = worldpoint_s['x'].iloc[wp_s_index]  #以帧号作为索引,wp(worldpoint),s(start),x(x坐标)
        self.wp_s_y = worldpoint_s['y'].iloc[wp_s_index] 
        wp_t_x = worldpoint_t['x'].iloc[wp_t_index]
        self.wp_t_y = worldpoint_t['y'].iloc[wp_t_index] 
        pp_index = posepoint[posepoint['croods']==crood_wp].index[0] #pp(posepoint) x(x坐标)
        pp_x = posepoint['x'].iloc[pp_index]
            #print(wp_s_x,wp_t_x,pp_x)
        return pp_x,wp_s_x,wp_t_x,pp_index,wp_s_index,wp_t_index
        '''
        except IndexError:
            #print('IndexError,未进入范围,继续循环')
            continue       
        '''


    
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
        for i in range(a, b):  
            #print('i===',i)
            crood0 = int(self.worldpoint3['croods'].iloc[0])
            #print(posepoint['croods'].iloc[i],crood0)
            if posepoint['croods'].iloc[i] < crood0: #从worldpoint3出现的时候开始进入循环
                continue
            crood_wp = int(posepoint['croods'].iloc[i]) #进入循环后，记录当前worldpoint的帧号
            #print('i=',i)
            #print('crood_wp=',crood_wp)
            #设置第一段起止点
            #print('correct_x=',self.correct_x)
            
            for d in range(2,5): #每个循环就是进入了每一段用不同的点矫正的轨迹中
                worldpoint_s = self.__dict__['worldpoint{}'.format(d)]
                worldpoint_t = self.__dict__['worldpoint{}'.format(d+1)]
                #print('假设移动到了点{}到点{}之间，i={},crood_wp={}'.format(d,d+1,i,crood_wp))
                if len(worldpoint_s[worldpoint_s['croods']==crood_wp]) == 1 : #如果进入循环的时候crood不在worldpoint_s范围内则跳过
                    pp_x, wp_s_x, wp_t_x, pp_index, wp_s_index, wp_t_index = self.create_crood_index(crood_wp, posepoint, worldpoint_s, worldpoint_t)
                    #print('crood在范围内',pp_x, wp_s_x, wp_t_x, pp_index, wp_s_index, wp_t_index)
                else:
                    #print('crood不在点{}范围内'.format(d),len(worldpoint_s[worldpoint_s['croods']==crood_wp]) == 1)
                    continue
                #print('wp_t_x={},pp_x={},wp_s_x={}'.format(wp_t_x,pp_x,wp_s_x))
                if wp_t_x < pp_x < wp_s_x: #用x轴的上下限来限定posepoint
                    
                    if crood_wp < start:
                        start = crood_wp
                    #print('动物已经经过wp3')
                    if worldpoint_s['x_d'].iloc[wp_s_index] > 100:
                        continue
                    #print('correct_x=',self.correct_x,wp_s_index,worldpoint_s['x_d'].iloc[wp_s_index])
                    self.correct_x = self.correct_x + (worldpoint_s['x_d'].iloc[wp_s_index] + worldpoint_t['x_d'].iloc[wp_s_index])*0.5 #累积计算correct_x相对位移
                    posepoint['x'].iloc[pp_index] = posepoint['x'].iloc[pp_index] - self.correct_x #修正posepoint
                    self.correct_y = self.correct_y + (worldpoint_s['y_d'].iloc[wp_s_index] + worldpoint_t['y_d'].iloc[wp_s_index])*0.5
                    posepoint['y'].iloc[pp_index] = posepoint['y'].iloc[pp_index] - self.correct_y
                    #print('XXX wp_t_x={},pp_x={},wp_s_x={}'.format(wp_t_x,pp_x,wp_s_x))
                    #print('移动到了点{}到点{}之间，i={},crood_wp={}'.format(d,d+1,i,crood_wp))
                    #print('correct_x={},correct_y={},dx = {}'.format(self.correct_x,self.correct_y, worldpoint_s['y_d'].iloc[wp_s_index]))
                    count_d = np.array([wp_t_x,self.wp_t_y]) - np.array([wp_s_x,self.wp_s_y])
                    k = math.hypot(count_d[0],count_d[1])/3
                    t = crood_wp
                    position_flag.append([crood_wp,d,k])
                    break  #这里的break很重要，因为在上一次的纠正中被改变的x值可能导致下个小循环中满足了另一个if的阈值，导致重复循环
            
                
        print('start',start) #这里标记了矫正是从第多少帧开始进行的
        return start,t
                #else:
                    #posepoint['x'].iloc[pp_index] = posepoint['x'].iloc[pp_index] - self.correct_x
                    #posepoint['y'].iloc[pp_index] = posepoint['y'].iloc[pp_index] - self.correct_y
                    #print('已过wp4')    
                    #pass



def get_xy(croods, point):
    #该函数用以获得某一时间croods的x，y坐标
    return  point.query('croods==@croods').x.iloc[0], point.query('croods==@croods').y.iloc[0]
def lr(croods, worldpdic):
    #根据6个worldpoint建立线性回归方程
    worldpoint_list = []
    for i in range(len(worldpdic)):
        p = worldpdic['worldpoint{}'.format(i+1)]
        if len(p.query('croods==@croods')) == 0: #如果该帧数不存在点，则跳过
            continue
        x,y = get_xy(croods, p)
        worldpoint_list.append([x,y])
    worldpoint_list = np.array(worldpoint_list)
    #plt.plot(worldpoint_list[:,0],worldpoint_list[:,1])
    #plt.scatter(worldpoint_list[:,0],worldpoint_list[:,1])
    #plt.show()
    point_num = len(worldpoint_list)
    dxy = worldpoint_list[0] - worldpoint_list[-1]
    l= math.hypot(dxy[0],dxy[1])
    k = (l/(point_num-1))/3
    
    '''
    lr_model = LinearRegression()  #导入一元线性模块
    lr_model.fit(worldpoint_list[:,0],worldpoint_list[:,1])  #进行拟合
    
    x2 = np.array(range(2000))  # 绘图验证
    y2 = lr_model.predict(x2)
    y2 = 1080 - y2
    plt.scatter(worldpoint_list[:,0],1080 - worldpoint_list[:,1]) 
    plt.plot(x2,y2)
    '''
    #plt.show()
    '''plot
    for i in range(400,1800,100):
    croods = i
    lr(croods, worldpdic)
    '''
    
    
    return k
#test k 
'''
list1 = []
for i in range(100,1500):
    list1.append([i,lr(i,worldpdic)])
list1
list1 = np.array(list1)
plt.scatter(list1[:,0],list1[:,1])
'''

class LinearRegression:
    """
    拟合一元线性回归模型

    Parameters
    ----------
    x : shape 为(样本个数,)的 numpy.array
        只有一个属性的数据集

    y : shape 为(样本个数,)的 numpy.array
        标记空间

    Returns
    -------
    self : 返回 self 的实例.
    """
    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, x, y):
        variance = np.var(x, ddof=1)  # 计算方差，doff为贝塞尔（无偏估计）校正系数
        covariance = np.cov(x, y)[0][1]  # 计算协方差
        self.w = covariance / variance
        self.b = np.mean(y) - self.w * np.mean(x)
#       self.w = np.sum(y * (x - np.mean(x))) / (np.sum(x**2) - (1/x.size) * (np.sum(x))**2)
#       self.b = (1 / x.size) * np.sum(y - self.w * x)
        return self

    def predict(self, x):
        """
        使用该线性模型进行预测

        Parameters
        ----------
        x : 数值 或 shape 为(样本个数,)的 numpy.array
            属性值

        Returns
        -------
        C : 返回预测值
        """
        return self.w * x + self.b




'''
————————————————————————————————————————————————————————————
接下来需要实现步态周期的计算和分割
————————————————————————————————————————————————————————————
方案1：可以用20-40格滑窗，每次都找到一个峰值。然后对每个峰值做判定，如果它大于前后
方案2：直接20格滑窗，找到范围内的最大值并标记，然后将标记相近的比大小合并。
'''  
def  peak_n(posepoint, start, n_crood=40):
    #从周围n个值中找到最大值并且加入某个DF中
    max_df = pd.DataFrame([])
    posepoint_max = posepoint.query('croods>@start')
    file_name(pose_path)
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
        max_series = posepoint_max.query('@start < croods < @ter').sort_values(by=['y']).iloc[-1]
        max_df = max_df.append(max_series)
    max_df.drop_duplicates(subset=['croods'],keep='first',inplace=True) #去重复
    max_df = max_df.sort_values(by = ['x'])
    
    maxpoint = pd.DataFrame([])
    #对筛选出的局部最高点进行合并，将距离在上下100范围内的合并起来
    print(max_df)
    
    
    max_df.sort_values(by=['croods'], inplace=True)
    #max_df.drop(max_df[max_df['x'] - max_df['x'].shift(1)>0].index)
    max_df.sort_values(by=['x'], inplace=True)
    while True:
        print(max_df)
        x = max_df['x'].iloc[0]
        targe = max_df.query('@x-100 < x < @x+120') #
        series = targe.sort_values(by=['y']).iloc[-1]
        maxpoint = maxpoint.append(series)
        
        max_df = max_df[len(targe):] #删掉已经被合并的数据    
    

        if len(max_df) == 0:
            break
    
    maxpoint.sort_values(by=['croods'], inplace=True)
    x_d = maxpoint['x'] - maxpoint['x'].shift(1)
    print(maxpoint[x_d>0].index)
    maxpoint = maxpoint.drop(maxpoint[x_d>0].index)
    #合并较小的cycle
    return maxpoint

def creat_idic(pose_path_fir, worldpoint_path_fir):
    #用于创建类的个体数据实例化
    idic = {}
    for i in range(len(pose_path_fir)): #len(pose_path_fir)
        idic['{}'.format(i)] = Sheep_individual()
        idic['{}'.format(i)].pretreatpoint(pose_path_fir[i],worldpoint_path_fir[i])
        point = idic['{}'.format(i)].lhoof
        p1 = point.copy()
        print(pose_path_fir[i].split('/')[-1].split('DLC')[0],'轨迹创建完毕,i=',i)
        try:
            start,position_flag = idic['{}'.format(i)].posepoint_correct(idic['{}'.format(i)].lhoof,0 ,len(point))
        except (AttributeError,KeyError):
            pass
        idic['{}'.format(i)].lhoof = idic['{}'.format(i)].lhoof.query('t>croods>@start')
        point = point.query('croods>@start')
        p1 = p1.query('croods>@start')
        plt.scatter(point['croods'],point['x'])
        plt.scatter(p1['croods'],p1['x'])    
        plt.show()

        
        #plot_scatter(S.b)    
        
    return idic

if __name__ == '__main__':
    '''
    #wordpoint from 
    '/home/hjh/DLC/XHrunning/T/XH_T_swing-hjh-2021-03-31'
    pose_path = '/home/hjh/DLC1/output/run/XH_jump_T/posepre/T57_022_1DLC_resnet50_XH_swing_TNov29shuffle6_900000.csv'
    worldpoint_path = '/home/hjh/DLC1/output/run/XH_jump_T/wordpoint/T57_022_1DLC_resnet50_XH_T_swingMar31shuffle1_250000.csv'
    #pose_df = pd.read_csv(pose_path,header=True,sep = ' ')
    '''
    #pose_path = '/home/hjh/DLC1/output/run/XH_jump_T/posepre'
    #pose_path = '/home/hjh/DLC1/output/run/XH_jump_T/oldversion_posepre'
    pose_path = '/home/hjh/DLC1/output/run/XH_jump_T/20220408run_posepre'
    worldpoint_path = '/home/hjh/DLC1/output/run/XH_jump_T/old'
    pose_path_fir = file_name(pose_path)
    pose_path_fir.sort()
    worldpoint_path_fir = file_name(worldpoint_path)
    worldpoint_path_fir.sort()

#创建类的实例化个体
    #S = Sheep_individual()
    idic = creat_idic(pose_path_fir, worldpoint_path_fir)

#前处理完成，开始后续分析
    p_class = idic['0']
    #刷新class
    p_class.pretreatpoint(pose_path_fir[15],worldpoint_path_fir[15])
    point = p_class.lhoof
    plt.scatter(point['croods'],point['x'])
    start = p_class.posepoint_correct(point,0 ,len(point))
    plt.scatter(point['croods'],point['x'])
    point = point.query('croods>@start')
    #point = S.bodypart2

    '''
dic_correct = {}
for i in range(len(idic)):
    p_class = Sheep_individual()
    p_class.pretreatpoint(pose_path_fir[i],worldpoint_path_fir[i])
    point = p_class.lhoof
    p1 = point.copy()
    
    try:
        start = p_class.posepoint_correct(point,0 ,len(point))
    except (AttributeError,KeyError):
        pass
    point = point.query('croods>@start')
    p1 = p1.query('croods>@start')
    plt.scatter(point['croods'],point['x'])
    plt.scatter(p1['croods'],p1['x'])    
    plt.show()
    dic_correct['{}'.format(i)] = point
    '''
    maxpoint = peak_n(point,start)
    maxpoint = maxpoint.sort_values(by=['croods'])


    crood_d = (maxpoint['croods'] - maxpoint['croods'].shift(1))/30
    maxpoint['cycle'] = crood_d 
    print(maxpoint[['croods','cycle']])
    
    
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
    '''
    del_index = [i for i in maxpoint.query('cycle<1.2').index]
    for i in range(1,len(maxpoint)-1):
        if maxpoint['cycle'].iloc[i] < 1.2: #如果一个周期小于1.2视频时长，则将其向前后合并
                
            if abs(maxpoint['croods'].iloc[i] - maxpoint['croods'].iloc[i-1]) <= abs(maxpoint['croods'].iloc[i] - maxpoint['croods'].iloc[i+1]):  #如果离上一个周期更近，就合并进去，否则合并进下一个周期
                maxpoint['cycle'].iloc[i-1] = maxpoint['cycle'].iloc[i] + maxpoint['cycle'].iloc[i-1]
            else:   
                maxpoint['cycle'].iloc[i+1] = maxpoint['cycle'].iloc[i] + maxpoint['cycle'].iloc[i+1]
            
    maxpoint = maxpoint.drop(del_index)
    '''

    '''
①T58纯黑个体识别有问题
②思考：是否应该用三阶样条插值代替拟合？
③可以尝试对比以各个点取局部极值的结果找步态周期（可以选一个比较稳定的点，也可以取集成的结果）
④可以合并连续的两个低于1.5s（真实时间0.25s）的点，或者对较长的cycle做删除和拆分
⑤关于第四个点的问题，应该是存在真实点，但是最高的那个是错误点，导致整个被否定
      croods     cycle
867   1442.0       NaN
948   1523.0  2.700000
1029  1604.0  2.700000
1116  1691.0  2.900000
1187  1762.0  2.366667
1245  1820.0  1.933333
1254  1829.0  0.300000
1330  1905.0  2.533333
1393  1968.0  2.100000

hind_lhoof
     croods     cycle
500  1488.0       NaN
608  1596.0  3.600000
688  1676.0  2.666667
754  1743.0  2.233333
827  1816.0  2.433333
937  1926.0  3.666667
976  1969.0  1.433333

hind_rhoof
     croods     cycle
258  1450.0       NaN
309  1510.0  2.000000
367  1595.0  2.833333
454  1687.0  3.066667
512  1751.0  2.133333
578  1825.0  2.466667
655  1909.0  2.800000
715  1988.0  2.633333

body3
      croods     cycle
1024  1442.0       NaN
1103  1521.0  2.633333
1186  1604.0  2.766667
1270  1688.0  2.800000
1344  1762.0  2.466667
1415  1833.0  2.366667
1504  1922.0  2.966667
1576  1994.0  2.400000


     croods     cycle
204   212.0       NaN
268   277.0  2.166667
347   356.0  2.633333
414   423.0  2.233333
482   498.0  2.500000
549   567.0  2.300000
609   631.0  2.133333
666   688.0  1.900000

    '''


    #f,m,n = S.worldpoint_pre(S.worldpoint2,order=6)
    #new_point = pd.DataFrame([f,m,n]).T
    '''
    p = S.hind_rhoof
    x = p['x']
    y = p['y']
    plt.scatter(x,y)
    plt.show
    '''
