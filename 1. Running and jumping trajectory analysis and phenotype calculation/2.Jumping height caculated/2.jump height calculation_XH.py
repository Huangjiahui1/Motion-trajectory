#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:25:40 2021

@author: hjh
"""
import numpy as np
import os
#import xlwt
import pandas as pd
import matplotlib.pyplot as plt
import regularexpression as re0  #导入自定义的正则表达式
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from numpy.polynomial.polynomial import Polynomial
from filt_pretreatpoint_tendst import * 
'''
————————————————————————————————————————————————————————————
命名识别
————————————————————————————————————————————————————————————
'''  
def name_order(path,sheet_name=0,I=0):

    '''
    sequence
    path:excel表的地址
    sheet_name:工作表的序号
    I:奔跑次数的序号
    '''
    seqdict = {}
    data = pd.read_excel(path, skiprows=0, usecols="A:U", dtype={'耳标1':str, '耳标2':str, '耳标3':str}, sheet_name=sheet_name)
    for i in range(1,4):    
        for j in range(len(data)):
            name = data['耳标%d' % i ][j]
            seqdict['S_{}_{}'.format(name, i)] = [data['起点%d' % i][j], data['喷漆%d' % i][j]]
            seqdict['M_{}_{}'.format(name, i)] = [data['中间%d' % i][j], data['喷漆%d' % i][j]]
            seqdict['T_{}_{}'.format(name, i)] = [data['终点%d' % i][j], data['喷漆%d' % i][j]]
    
    return seqdict

  
#对比下降
'''
————————————————————————————————————————————————————————————
进行多项式拟合 
————————————————————————————————————————————————————————————
'''  
def binfit(new,s,t,order,plot=False,flag_nonereal=0,individualname=''):
    
    #构建x和y多项式拟合
    x = []
    y = []
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    
    
#    s,t = tendst(new)
    if (t - s > 50):          
        for i in range(len(new)):
            x0.append(new[i][1])
            y0.append(new[i][2])
            if t >= new[i][0] >= s:
                x.append(new[i][1])
                y.append(new[i][2])
    else:
        print('代码错误，请修改')
        x = x1
        y = y1
    try:
        minpoint = max(y)
    except ValueError:
        pass
    c = []
    for i in y:
        c.append(i)
    
            
    x = np.array(x)
    y = np.array(y)
    x2 = np.linspace(x[-1],x[0])
    #多项式拟合
    params = np.polyfit(x,y,order) #拟合
    funcs = np.poly1d(params) #funcs为拟合函数
    ypre = funcs(x) #用拟合函数和x值来预期y值
    if plot:     #*******************第一张图
        #是否画图
        if flag_nonereal ==1:
            plt.scatter(x,y,color='red')
            plt.plot(x2,funcs(x2),color='red')
            plt.savefig(r'F:\data\jump_model\{}_pre_e.jpg'.format(individualname))
            plt.show()
        else:
            plt.scatter(x,y) #原散点图
            plt.plot(x2,funcs(x2)) #拟合曲线图
            plt.savefig(r'F:\data\jump_model\{}_pre.jpg'.format(individualname))
            plt.show()

    
    P = funcs  # 多项式系数 数组
    x1 = np.linspace(0, 1920)
    y1 = np.polyval(P, x1)  # 求多项式 Y 数组
    #plt.plot(x, y)
    
    # 求导函数
    Q = np.polyder(P)  # 求导数
    xs = np.roots(Q)  # 求多项式函数根
    
    xs0 = []
    for i in xs:
        
        if x[0] <= i <= x[-1]:   #************此处因为玛曲和夏河两地奔跑方向相反而有不同**********
        #if x[-1] <= i <= x[0]:
            xs0.append(i)
    xs = np.array(xs0)
    ys = np.polyval(P, xs)
    
    # 绘制曲线拐点
    #plt.scatter(xs, ys, color="r", s=60)
    real = []
    for i in ys:
        im = i.imag
        if im == 0:
            real.append(i.real)
    print(real)
    try:
        miny = max(real)
    except :
        flag_nonereal = 1
        print('无拐点、根')

#    plt.show()
#    plt.scatter(x1,y1)
#    plt.show()
    if plot:#*************************第二张图
        if flag_nonereal ==1:
            plt.scatter(x0,y0,color='red')
            plt.savefig(r'F:\data\jump_model\{}_e.jpg'.format(individualname))
            #print('已保存——————————————————————————————————')
            plt.show()
            
        else:
            plt.scatter(x0,y0)
            plt.savefig('xx.jpg')
            plt.savefig(r'F:\data\jump_model\{}.jpg'.format(individualname))
            #print('——————已保存——————————————————————————————————')
            plt.show()
            
    return x,y,ypre,funcs,miny,minpoint,ys,xs

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
        '''
        #XH
        if (t <= new[i][0] <= 1900) and ((550 <= new[i][2] <= 850)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])     
        if (50 <= new[i][0] <= s) and ((550 <= new[i][2] <= 850)):
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])
        '''
        '''
        if (t <= new[i][0] <= t+30) :#and ((550 <= new[i][2] <= 850))
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])
        '''
        if (s-30 <= new[i][0] <= s): #and ((550 <= new[i][2] <= 850))
            #print(s,s-50,new[i][0])
            avglist.append(new[i][2])
        
        else:
            continue    
    #print(avglist)
    avg = np.mean(avglist)
    print('地面高度为：{}'.format(avg))
    return avg
'''
def high1(new,s=0,t=0):
    #**********************根据不同缩放的模型需要有所改变*****************************
    avglist = []
    newf = pd.DataFrame(new)
    for i in range(len(newf)):  
        if (t <= new[i][0] <= t+50) :#and ((560 <= new[i][2] <= 640))
            avglist.append(new[i][2])
        else:
            continue    
    avg = np.mean(avglist)
    print('地面高度为：{}'.format(avg))
    return avg
'''
'''
————————————————————————————————————————————————————————————
打印多项式
————————————————————————————————————————————————————————————
'''  
class Polynomial(list): 
    #多项式打印（get自网上）
    def __repr__(self):
        # joiner[first, negative] = str
        joiner = {
            (True, True): '-',
            (True, False): '',
            (False, True): ' - ',
            (False, False): ' + '
        }

        result = []
        for power, coeff in reversed(list(enumerate(self))):
            j = joiner[not result, coeff < 0]
            coeff = abs(coeff)
            if coeff == 1 and power != 0:
                coeff = ''

            f = {0: '{}{}', 1: '{}{}x'}.get(power, '{}{}x^{}')

            result.append(f.format(j, coeff, power))

        return ''.join(result) or '0'
    
'''
————————————————————————————————————————————————————————————
路径和文件名
————————————————————————————————————————————————————————————
'''  
#pre = '/home/hjh/DLC1/output/jump/XH_jump/XH_jump_10'
#pre = '/home/hjh/DLC1/output/jump/MQ_jump/MQ_jump_12' 
pre = r'F:\softln\DLC1\output\jump_2.2C\XH_jump\XH_jump_7.5'
pre = r'F:\softln\DLC1\output\jump_2.2C\XH_jump\XH_jump_10'
def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            
            allname = file_dir + '\\' + file   #it's a difference between linux and win
            L.append(allname)
    return L

'''
————————————————————————————————————————————————————————————
MODEL打包 
————————————————————————————————————————————————————————————
'''  
'''
————————————————————————————————————————————————————————————
旋转工作的文件名和列表准备
————————————————————————————————————————————————————————————
'''  
#为了让特定转动的一些视频旋转回正，这一块用于旋转
r_path = r'F:\softln\DLC1\output\jump\MQ_jump\MQ_jump_6\顺时针'
r_path_1 = r'F:\softln\DLC1\output\jump\MQ_jump\MQ_jump_12\S'
l_path = r'F:\softln\DLC1\output\jump\MQ_jump\MQ_jump_6\逆时针'

right_spin_list = file_name(r_path)
right_spin_list_1 = file_name(r_path_1)
left_spin_list = file_name(l_path)
rlist = []
llist = []
for i in range(len(right_spin_list)):
    name_r = right_spin_list[i].split('/')[-1]
    rlist.append(name_r)
for i in range(len(right_spin_list_1)):
    name_r = right_spin_list_1[i].split('/')[-1]
    rlist.append(name_r)    

for i in range(len(left_spin_list)):
    name_l = left_spin_list[i].split('/')[-1]
    llist.append(name_l)




def find_best_model_and_max_y(x, y, degrees=[2, 3, 4, 5]):
    weights = np.ones_like(x)
    midpoint = int(len(x) / 2)
    weights[:midpoint] = np.linspace(1, 10, midpoint)
    weights[midpoint:] = np.linspace(10, 1, len(x) - midpoint)

    best_degree = None
    best_mse = float('inf')
    best_model = None
    best_max_y = None
    best_max_x = None

    for degree in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree)),
            ('linear', LinearRegression())
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = cross_val_score(model, x[:, np.newaxis], y, scoring='neg_mean_squared_error', cv=kf, fit_params={'linear__sample_weight': weights})
        average_mse = -np.mean(mse_scores)

        model.fit(x[:, np.newaxis], y, linear__sample_weight=weights)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = model.predict(x_smooth[:, np.newaxis])

        max_y = np.max(y_smooth)
        max_x = x_smooth[np.argmax(y_smooth)]

        if average_mse < best_mse:
            best_mse = average_mse
            best_degree = degree
            best_model = model
            best_max_y = max_y
            best_max_x = max_x

    return best_degree, best_model, best_max_y, best_max_x, best_mse

def evaluate_model_in_range(model, s, t):
    # Extract polynomial coefficients
    poly_features = model.named_steps['poly']
    linear_model = model.named_steps['linear']
    # Get the full array of coefficients for the fitted model
    coeffs = linear_model.coef_
    intercept = linear_model.intercept_
    # Adjust the coefficients with intercept properly placed
    coeffs = np.concatenate(([intercept], coeffs[1:]))

    # Create a polynomial from coefficients
    p = Polynomial(coeffs)

    # Evaluate polynomial at a fine grid within the range
    x_vals = np.linspace(s, t, 500)
    y_vals = p(x_vals)

    # Find maximum value and its corresponding x
    max_y = np.max(y_vals)
    max_x = x_vals[np.argmax(y_vals)]

    # Calculate first and second derivatives
    first_derivative = p.deriv(1)
    second_derivative = p.deriv(2)

    # Find roots of the first derivative (critical points)
    critical_points = first_derivative.roots()
    valid_critical_points = critical_points[(critical_points >= s) & (critical_points <= t) & np.isreal(critical_points)]
    valid_critical_points = valid_critical_points.real

    # Check if maximum value's x is a root of the first derivative
    is_max_x_a_root = np.isclose(valid_critical_points, max_x, atol=1).any()

    # Check the number of valid critical points
    #if len(valid_critical_points) != 1:
    #    raise ValueError("Number of critical points is not equal to one.")

    # Filter roots of second derivatives to be within the range and real (inflection points)
    inflection_points = second_derivative.roots()
    valid_inflection_points = inflection_points[(inflection_points >= s) & (inflection_points <= t) & np.isreal(inflection_points)]
    valid_inflection_points = valid_inflection_points.real

    print(f"Maximum Y: {max_y} at X: {max_x}")
    print(f"Critical Points (Potential Extrema) within range: {valid_critical_points}")
    print(f"Inflection Points within range: {valid_inflection_points}")
    print(f"Is maximum X a root of the first derivative? {'Yes' if is_max_x_a_root else 'No'}")

    return max_x, max_y, valid_critical_points, valid_inflection_points, is_max_x_a_root

def benifit_v5(new,s,t,individualname=''):
    # 初始化一个空列表来存储模型的交叉验证得分
    #构建x和y多项式拟合
    x = []
    y = []
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    #    s,t = tendst(new)
    if (t - s > 50):          
        for i in range(len(new)):
            x0.append(new[i][1])
            y0.append(new[i][2])
            if t >= new[i][0] >= s:
                x.append(new[i][1])
                y.append(new[i][2])
    else:
        print('代码错误，请修改')
        x = x1
        y = y1
    try:
        maxpoint = max(y)
    except ValueError:
        pass
    c = []
    for i in y:
        c.append(i)
        
                
    x = np.array(x)
    y = np.array(y)
    
    best_degree, best_model, best_max_y, best_max_x, best_mse = find_best_model_and_max_y(x, y)
    #求实根验证
    max_x, max_y, valid_critical_points, valid_inflection_points, is_max_x_a_root = evaluate_model_in_range(best_model, x[0], x[-1])
    print(f"Best Model Degree: {best_degree}")
    print(f"Maximum Y: {best_max_y} at X: {best_max_x}")
    print(f"Best Model MSE: {best_mse}")
    print(f'real root：{valid_critical_points}')
    # 可视化最佳拟合模型
    '''
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='gray', label='Original Data', alpha=0.5)
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = best_model.predict(x_smooth[:, np.newaxis])
    plt.plot(x_smooth, y_smooth, label=f'Best Fit Degree {best_degree}')
    plt.scatter([best_max_x], [best_max_y], color='red', label=f'Maximum Y at X={best_max_x:.2f}')
    plt.title('Best Polynomial Fit with Cross-Validation')
    plt.xlabel('{}-X'.format(individualname))
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(r'F:\项目数据\运动GWAS项目绘图\附图\jump\{}'.format(individualname))
    plt.show()
    '''
    if not is_max_x_a_root:
        raise ValueError("root error")
    
    
    return x, y, best_max_x, best_max_y, maxpoint, best_degree, best_model



def model(path, wind=12, order=6,n_neighbors=10,individual = 2,OD = 'none', flag_nonereal=0, plot=False, individualname='',llist=llist,rlist=rlist):
    """
    path:路径 str
    wind:起跳识别窗口大小 int
    order:多项式阶数 int
    ken:视频的视野范围你
    """
    
    new = pretreatpoint(path,individual=individual)

    if path.split('\\')[-1] in rlist:
        print('视频顺时针旋转1.5')
        new = spin.spin_r(new, 1.5)
    elif path.split('\\')[-1] in llist:
        print('视频逆时针旋转3.1')
        new = spin.spin_l(new, 3.1)
    else:
        print('仅倒置，不旋转')
        new = spin.spin_r(new, 0)
    #print(o)
    print(new)
    if individualname == 'M_015_2':
        new = new[new[:,1]>600]
        
                
    s,t,minp_average = tendst(new,wind)#默认为point2，即第二个点
    #o,R,MSElist,minMSE = model_test(new,s,t)    #画图并且导出多项式
    #normal_x,normal_y,ypre,funcs,miny,minpoint,ys,xs = binfit(new,s,t,order=o,plot=plot,flag_nonereal=flag_nonereal,individualname=individualname)
    normal_x, normal_y, best_max_x, miny, maxpoint, best_degree, best_model = benifit_v5(new,s,t,individualname=individualname)
    #求实根验证

    avgy = high(new,s,t)
    dy = miny - avgy
    #print(funcs)
    
    distance = normal_x[-1] - normal_x[0]
    
    print('抬升高度：' + str(dy) + '最高拟合点' + str(miny) + '最高像素点:'+str(maxpoint))
#    print("多项式:"+str(Polynomial(funcs)))
#    print('最小值为'+str(miny))
    return miny,dy,maxpoint,avgy,distance,minp_average,normal_x,normal_y,best_max_x,miny,best_model
'''        
    for i in range(len(point)):
        ts = float(point[s][2]) - float(point[i][2])
        if ts < 50:
            continue
'''

def model_plot(x,y,best_max_x,best_max_y,best_model,individualname=''):
    # 可视化最佳拟合模型
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='gray', label='Original Data', alpha=0.5)
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = best_model.predict(x_smooth[:, np.newaxis])
    plt.plot(x_smooth, y_smooth, label=f'Best Fit Degree {best_degree}')
    plt.scatter([best_max_x], [best_max_y], color='red', label=f'Maximum Y at X={best_max_x:.2f}')
    plt.title('Best Polynomial Fit with Cross-Validation')
    plt.xlabel('{}-X'.format(individualname))
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(r'F:\项目数据\运动GWAS项目绘图\附图\jump\XH_10M\{}'.format(individualname))
    plt.show()
#区分两条轨迹并命名

if __name__ == '__main__':
    
    L = file_name(pre)
    q = 1
    p = 1
    
    result1 = []
    '''
    ——————————————————————————————————————————————————————————————————————————
    制作一个编号字典  
    ——————————————————————————————————————————————————————————————————————————
    '''  
    where = re.search(r'(MQ)|(XH)', pre)
    sheet_name = 2  ######################3代表夏河跳高，2代表玛曲跳高
    if where.group() == 'XH':
        sheet_name = 3
    #namef = {}
    #path1 = r'D:\QQ\1042015037\FileRecv\视频质量评估20210813夏河全纠正.xlsx'
    path1 = r'F:\项目数据\视频质量评估20210813夏河全纠正.xlsx'
    namef = name_order(path1,sheet_name=sheet_name)
        
    '''x = 
    ——————————————————————————————————————————————————————————————————————————
    运行、调参
    ——————————————————————————————————————————————————————————————————————————len(down)
    '''  
    ct = 0   #统计在下面这一步的高度计算中有通过的个体数目
    high1 = []
    c = 1 # 统计lack出现的次数  len(L)
    for d in range(len(L)):     #**************************************************************************************************
        if L[d][-3:] != 'csv':
            continue    
        miny1,dy1,maxpoint1,avgy1,distance1,minp_average1,x1,y1,best_max_x1,best_max_y1,best_model1 = [0]*11
        miny2,dy2,maxpoint2,avgy2,distance2,minp_average2,x2,y2,best_max_x2,best_max_y2,best_model2 = [0]*11
        

        rsearch = re0.Re0()  #导入自定义的正则表达式模块
        name = rsearch.name(L[d]) 
        if name == None:
            continue 
        print(name)
        first = 1
        namelist = name.split("_")
        name1 = namelist[0][0] + '_' + namelist[1] + '_' + namelist[-1]
        if len(namelist) > 3:
            name2 = namelist[0][0]  + '_' + namelist[3] + '_' + namelist[-1]
        else:
            name2 = 0

        '''
    ——————————————————————————————————————————————————————————————————————————
    给跳跃命名编号
    ——————————————————————————————————————————————————————————————————————————
        '''  
        wind = 12
        flag_nonereal = 0
        tag = '1'
        miny1 = 0
        miny2 = 0
        try:
            miny1,dy1,maxpoint1,avgy1,distance1,minp_average1,x1,y1,best_max_x1,best_max_y1,best_model1 = model(L[d],wind=wind,n_neighbors=10,OD = 'none',individual=1,individualname=name1)
        except (ValueError,IndexError):
            flag_nonereal_1=1
            print('{}视频的1号个体无结果'.format(L[d]))
            
            dy1 = 0
            tag = 'lack%d' % c#当有数据损失时做出标记
            c += 1
            ct += 1
        try:

            miny2,dy2,maxpoint2,avgy2,distance2,minp_average2,x2,y2,best_max_x2,best_max_y2,best_model2 = model(L[d],wind=wind,n_neighbors=10,OD = 'none',individual=2,individualname=name2)
        except (ValueError,IndexError):
            flag_nonereal_2=1
            print('{}视频的2号个体无结果'.format(L[d]))
            
            dy2 = 0
            ct += 1
            tag = 'lack%d' % c#当有数据损失时做出标记
            c += 1
        if miny1 != 0 and miny2 != 0:
            if minp_average1 >= minp_average2:
                miny1,miny2 = miny2,miny1 
                dy1,dy2 = dy2,dy1

                mp1,mp2 = mp2,mp1
                mpdy1,mpdy2 = mpdy2,mpdy1
                avg1,avg2 = avg2,avg1
        if miny1 == 0 and miny2 != 0:
            name2 = name1
        try:
            model_plot(x1, y1, best_max_x1, best_max_y1, best_model1, name1)
        except AttributeError:
            pass
        try:
            model_plot(x2, y2, best_max_x2, best_max_y2, best_model2, name2)
        except AttributeError:
            pass
        
        print(name1,name2,miny1,miny2)
        if name1 == 0:
            namef[name1] = [0,0]
        if name2 == 0:
            namef[name2] = [0,0]
        namef[name1] = namef[name1]
        namef[name2] = namef[name2]
        
        high1.append(avgy1)
        high1.append(avgy2)
        ''' 
    ——————————————————————————————————————————————————————————————————————————
    klist:记录着像素坐标系与世界坐标系比例的字典
    ——————————————————————————————————————————————————————————————————————————
        '''  #MQ_12M中，T终点k值为179，起点K值为170
        klist = {'MQ_jump_12':179.32,
                 'MQ_jump_6_S':170,
                 'MQ_jump_6':304.24,
                 'MQ_jump_6_L':350,
                 'MQ_jump_4.5':464.08,
                 'XH_jump_10':197.07,
                 'XH_jump_7.5':259.77,
                 'XH_jump_5':407.15,
                 'MQ_jump_3.44':591,
                 'MQ_jump_7.5':240
                 }
        size = pre.split('\\')[-1]
        #6M中有一部分是不到6M的，为中间位置所拍摄，k值为350
        k = klist[size]
        '''
        #仅玛曲启用
        if '6/M' in L[d]:
            k = klist['MQ_jump_6_L']
        if '12/S' in L[d]:
            k = klist['MQ_jump_6_S']        
        '''
        '''
        mp:全局最小的识别点纵坐标
        mpdy:mp与动物站立高度值的差，为动物跳跃过程中的抬升高度
        '''
        try:
            dy1 = dy1/k #除以像素点转换比例比例k
            dy2 = dy2/k
            mpdy1 = mpdy1/k
            mpdy2 = mpdy2/k
            #if name1 == 0:
            #    name1 = '0'
            #if name2 == 0:
            #    name2 = '0'
                
            q1 = namef[name1] #视频质量和喷漆色组成的列表，例如：['A*','绿']
            q2 = namef[name2]
            if q2 == [0,0]:
                result1.append([name1,miny1,dy1,mp1,mpdy1,tag,q1[0],q1[1]])
                result1.append([name1,0,0,0,0,0,0,0])
                continue
            
            if miny1 ==0 or miny2 == 0:
                if   miny1 == 0 and miny2 == 0:
                    result1.append([name1,0,0,mp1,mpdy1,tag,q1[0],q1[1]])
                    result1.append([name2,0,0,mp2,mpdy2,tag,q2[0],q2[1]])
                    continue
                elif miny1 == 0 and miny2 !=0:
                    result1.append([name1,0,0,mp1,mpdy1,tag,q1[0],q1[1]])
                    result1.append([name2,miny2,dy2,mp2,mpdy2,tag,q2[0],q2[1]])
                    continue
                elif miny2 == 0 and miny1 !=0:
                    result1.append([name1,miny1,dy1,mp1,mpdy1,tag,q1[0],q1[1]])
                    result1.append([name2,0,0,mp2,mpdy2,tag,q2[0],q2[1]])
                    continue

            if '*' in q1[0]:  #如果优先标记在name1上，则执行
                result1.append([name1,miny1,dy1,mp1,mpdy1,tag+'*',q1[0],q1[1]])
                result1.append([name2,miny2,dy2,mp2,mpdy2,tag,q2[0],q2[1]])
                continue
            elif '*' in q2[0]:  #如果优先标记在name2上，则反过来
                result1.append([name2,miny1,dy1,mp1,mpdy1,tag+'*',q2[0],q2[1]])
                result1.append([name1,miny2,dy2,mp2,mpdy2,tag,q1[0],q1[1]])
                continue
            else:
                result1.append([name1,miny1,dy1,mp1,mpdy1,'none*',q1[0],q1[1]])
                result1.append([name2,miny2,dy2,mp2,mpdy2,'none*',q2[0],q2[1]])
        except KeyError:
            ct += 1
            pass

    result_df = pd.DataFrame(result1)
    result_df.columns = ['name','miny','dy','mp','mpdy','tag','q1','q2']
    #result_df.to_excel('F:\项目数据\运动GWAS项目绘图\附图\DLCXH_jump_7.5_20240604_1.xlsx')
    #result_df.to_excel('F:\项目数据\运动GWAS项目绘图\附图\DLCXH_jump_10_20240605.xlsx')
            #提取打包

    #写入excel表格
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet")
    sheet.write(0, 0, 'name')
    sheet.write(0, 1, 'miny')
    sheet.write(0, 2, 'dy')
    sheet.write(0, 3, 'mp')
    sheet.write(0, 4, 'mpdy')
    sheet.write(0, 5, 'tag')
    sheet.write(0, 6, 'q1')
    sheet.write(0, 7, 'q2')

    
    for i in range(1,len(result1)+1):
        for j in range(len(result1[i-1])):
            sheet.write(i, j, str(result1[i-1][j]))
    
    workbook.save(r"F:\项目数据\DLC-GWAS文章整理\DLCXH_jump_7.5(20230911).xlsx") 
    #workbook.save("/home/hjh/DLC1/output/jump/MQ_jump_12_20211029.xlsx")    
    #print(L[0])