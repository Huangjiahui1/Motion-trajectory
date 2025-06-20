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
import argparse

# 定义一个类用来存放个体的运动数据,以个体为单位
def plot_scatter(x, y, order):
    params = np.polyfit(x, y, order)
    funcs = np.poly1d(params)
    ypre = funcs(x)
    plt.plot(x, ypre)
    plt.scatter(x, y)
    plt.show()


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            allname = file_dir + '/' + file  # it's a difference between linux and win
            L.append(allname)
    return L


def file_name_k(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            allname = root + '/' + file  # it's a difference between linux and win
            L.append(allname)
    return L


# 按照不同大小的缩放尺寸视频进行分类
def k_name(k_l, name):
    # to research the k value
    for n, klist in k_l.items():
        if name in klist:
            return float(n)


'''
————————————————————————————————————————————————————————————
Sheep_individual这个类中包含了①文件读取、②点的过滤、③worldpoint的拟合填充
————————————————————————————————————————————————————————————
'''


class Sheep_individual:
    def __init__(self):
        self.correct_x = 0
        self.correct_y = 0
        pass

    def _creatpoint(self, pose_data, worldpointflag=False):
        # 输入为pandas的DataF格式
        # 类外批量添加变量
        pointlist = {}
        for i in range(1, len(pose_data.columns), 3):
            x = [pose_data.columns[0], pose_data.columns[i], pose_data.columns[i] + '.1', pose_data.columns[i] + '.2']
            pointlist[pose_data.columns[i]] = pose_data[x][1:]  # 删除重复的第一行
            pointlist[pose_data.columns[i]].columns = ['croods', 'x', 'y', 'likelihood']  # 所有变量改标题
            pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].astype(float)  # 全体变量改浮点数
            pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].query('likelihood>0.6')  # 根据likelihood >0.8过滤
            pointlist[pose_data.columns[i]] = pointlist[pose_data.columns[i]].reset_index(drop=True)
            setattr(self, pose_data.columns[i], pointlist[pose_data.columns[i]])  # 用于给类添加属性的setattr方法
        return pointlist

    def pretreatpoint(self, pose_path):
        '''
        individual:1
        bodypart bodypart2 mid bodypart3 objectA b6	b7 b8 lhoof	rhoof hind_lhoof hind_rhoof
        此处用于输入数据的预处理
        '''
        pose_data = pd.read_csv(pose_path, keep_default_na=False, header=1)
        # 根据pose_data中所有识别的点自动创建变量，每个变量对应的数据以 ['crood','x','y','likelihood']格式
        self.posepdic = self._creatpoint(pose_data)  # 生成储存点轨迹的字典文件


'''
————————————————————————————————————————————————————————————
接下来需要实现步态周期的计算和分割
————————————————————————————————————————————————————————————
方案：直接40格滑窗，找到范围内的最大值并标记，然后将标记相近的比大小合并。
'''


class Cycle_count_peak:
    def __init__(self):
        pass

    def peak_n(self, posepoint, start, n_crood=40):
        # 从周围n个值中找到最大值并且加入某个DF中
        max_df = pd.DataFrame([])
        posepoint_max = posepoint.query('croods>@start')

        for i in range(len(posepoint_max)):
            crood = posepoint_max[i:i + 1]['croods'].iloc[0]
            if crood - n_crood < posepoint_max[:1]['croods'].iloc[0]:
                start = 0
            else:
                start = crood - n_crood
            if crood + n_crood > posepoint_max[-1:]['croods'].iloc[0]:
                ter = posepoint_max[-1:]['croods'].iloc[0]
            else:
                ter = crood + n_crood
            if len(posepoint_max.query('@start < croods < @ter')) == 0:
                continue
            max_series = posepoint_max.query('@start < croods < @ter').sort_values(by=['y']).iloc[-1]
            max_df = pd.concat([max_df, max_series.to_frame().T], ignore_index=True)
        max_df.drop_duplicates(subset=['croods'], keep='first', inplace=True)  # 去重复
        max_df = max_df.sort_values(by=['x'])
        maxpoint = pd.DataFrame([])
        # 对筛选出的局部最高点进行合并，将距离在上下100范围内的合并起来
        print(max_df)

        max_df.sort_values(by=['croods'], inplace=True)
        while True:
            print(max_df)
            x = max_df['x'].iloc[0]
            targe = max_df.query('@x-100 < x < @x+120')
            series = targe.sort_values(by=['y']).iloc[-1]
            maxpoint = pd.concat([maxpoint, series.to_frame().T], ignore_index=True)
            max_df = max_df[len(targe):]  # 删掉已经被合并的数据
            if len(max_df) == 0:
                break

        maxpoint.sort_values(by=['croods'], inplace=True)
        x_d = maxpoint['x'] - maxpoint['x'].shift(1)
        print(maxpoint[x_d > 0].index)
        maxpoint = maxpoint.drop(maxpoint[x_d > 0].index)
        # 合并较小的cycle
        return maxpoint

    def maxpoint_filt(self, point, maxpoint, name):
        maxpoint = maxpoint.sort_values(by=['croods'])
        crood_d = (maxpoint['croods'] - maxpoint['croods'].shift(1)) / 30
        maxpoint['cycle'] = crood_d
        print(maxpoint[['croods', 'cycle']])
        # 过滤较小的cycle，靠近的选更高的
        for i in range(1, len(maxpoint) - 1):
            if maxpoint['cycle'].iloc[i] < 1.2:  # 如果一个周期小于1.2视频时长，则将其向前后合并。视频1.2s等于实际时间0.2s

                if abs(maxpoint['y'].iloc[i] > maxpoint['y'].iloc[i - 1]):  # 如果离上一个周期更近，就合并进去，否则合并进下一个周期
                    maxpoint['cycle'].iloc[i] = maxpoint['cycle'].iloc[i] + maxpoint['cycle'].iloc[i - 1]
                    maxpoint['cycle'].iloc[i - 1] = 0
                else:
                    maxpoint['cycle'].iloc[i - 1] = maxpoint['cycle'].iloc[i] + maxpoint['cycle'].iloc[i - 1]
                    maxpoint['cycle'].iloc[i] = 0
            # 排除假阳性的点,以局部最高点在point集合中周围支持的点个数大于15个为标准
        maxpoint = maxpoint.query('cycle!=0')
        for i in range(len(maxpoint)):
            p = maxpoint['x'].iloc[i]
            if len(point.query('@p-100<x<@p+100')) < 15:
                maxpoint['cycle'].iloc[i] = 0
        maxpoint = maxpoint.query('cycle!=0')
        ax1.scatter(1960 - point['x'], point['y'])
        ax1.scatter(1960 - maxpoint['x'], maxpoint['y'])
        ax1.set_xlabel('x-coordinate')
        ax1.set_ylabel('y-coordinate')
        # plt.savefig(r'F:\项目数据\运动GWAS项目绘图\cycle\{}_x-y.pdf'.format(name))
        # plt.show()
        return maxpoint


def creat_idic(pose_path_fir):
    # 用于创建类的个体数据实例化
    idic = {}
    for i in range(len(pose_path_fir)):  # len(pose_path_fir) #第一个文件有格式问题
        print(pose_path_fir[i])
        idic['{}'.format(i)] = Sheep_individual()
        idic['{}'.format(i)].pretreatpoint(pose_path_fir[i])
        # plot_scatter(S.b)
        print(pose_path_fir[i].split('/')[-1].split('DLC')[0], '轨迹创建完毕,i=', i)
    return idic


class Cycle_count_touchtime:
    def _cycle_count(self, point, seppoint):
        cyclelist = []
        seppoint = pd.DataFrame(seppoint)
        for i in range(len(seppoint)):
            if i == 0:
                t = seppoint.iloc[i + 1][0]
                s = seppoint.iloc[i][0]
                cyclelist.append(point.query('@s<=croods<@t'))
            elif i == len(seppoint) - 1:
                s = seppoint.iloc[i][0]
                cyclelist.append(point.query('croods>=@s'))
            else:
                s = seppoint.iloc[i][0]
                t = seppoint.iloc[i + 1][0]
                cyclelist.append(point.query('@s<=croods<@t'))
        return cyclelist

    def _get_cycle(self, point):  # Used to divide the period
        point['d'] = point['x'] - point['x'].shift(1)
        stable_list = []
        range_list = []
        s = 0
        for i in range(0, len(point)):
            # print(cycle.croods.iloc[i],abs(cycle[i:i+5].d.sum()))
            if abs(point[i:i + 5].d.sum()) < 10:
                stable_list.append(point.croods.iloc[i])
        stable_df = pd.DataFrame(stable_list)
        stable_df['d'] = stable_df - stable_df.shift(1)
        d_point = stable_df.query('d>10')
        a = 0
        b = 0
        for i in range(len(d_point)):
            if i == 0:
                b = d_point.index[0] - 1
                range_list.append([stable_df.iloc[0][0], stable_df.iloc[b][0]])
            else:
                a = d_point.index[i - 1]
                b = d_point.index[i] - 1
                range_list.append([stable_df.iloc[a][0], stable_df.iloc[b][0]])
        a = d_point.index[-1]
        range_list.append([stable_df.iloc[a][0], stable_df.iloc[-1][0]])
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
        for range1 in range_list:  # 用peak峰值方法算出的结果结合验证,得到一组较为准确的触地时期。用触地时期计算步态周期
            # 用取并集的方法，如果一个range没有被maxpoint选中，并且其长度小于一定值（15），则舍弃掉
            a, b = range1
            if abs(a - b) >= 8:  # pass
                print([a, b])
                self.stable_range_list.append([a, b])
                continue
            for i in range(len(maxpoint)):
                if maxpoint.croods.iloc[i] in [i for i in range(int(a), int(b))]:
                    self.stable_range_list.append([a, b])

        stable_start_array = np.array(self.stable_range_list)[:, 0]
        cycle_range_list = []
        for i in range(len(stable_start_array)):
            if i == len(stable_start_array) - 1:
                cycle_range_list.append([stable_start_array[i], self.stable_range_list[-1][-1]])
            else:
                cycle_range_list.append([stable_start_array[i], stable_start_array[i + 1]])
            print('cycle_range_list:', cycle_range_list)
        self.cycle_range_list = cycle_range_list
        return stable_start_array

    # 用cycle_count分割步态周期##########
    '''
    ————————————————————————————————————————————————————————————
    用左前肢触地步态周期作为划分标准，用cycle_count分割步态周期
    ————————————————————————————————————————————————————————————
    '''

    def create_dict(self, stable_start_array, name_indv):
        self.limbs_list = ['lhoof', 'rhoof', 'hind_lhoof', 'hind_rhoof']
        self.cycle_dict = {}
        for name in self.limbs_list:
            point_n = p_class.__dict__[name]
            self.cyclelist = self._cycle_count(point_n, stable_start_array)
            self.cycle_dict[name] = self.cyclelist
        # plot
        # q = 221
        # for cyclelist in self.cycle_dict.values():
        #    plt.subplot(q)
        #    q += 1
        #    for cycle in cyclelist[:]:
        #        plt.scatter(cycle['croods'],cycle['x'])
        # plt.show()
        # plot 1
        for cyclelist in self.cycle_dict.values():
            for cycle in cyclelist[:]:
                ax2.scatter(1960 - cycle['x'], cycle['croods'])

            break
        ax2.set_xlabel('x-coordinate')
        ax2.set_ylabel('frames')

        # plt.savefig(r'F:\项目数据\运动GWAS项目绘图\cycle\{}_frames-x.pdf'.format(name_indv))
        # plt.show()

        return self.cycle_dict

    # 计算触地时间以lhoof为标准
    # 找到时序数据的平稳位置，以计算触地时间
    def _stable_time(self, cycle):  # use to count the others cycle #######################
        cycle['d'] = cycle['x'] - cycle['x'].shift(1)
        stable = []
        d10 = cycle.query('d>10')
        x10 = cycle.query('d<-10')
        if len(d10 > 0):  # 清除来回振动,找到一个d>10的最大差值点，进一步寻找它附近的反向振动
            for _max in d10.sort_values(by=['d']).index:
                try:
                    _min = cycle.loc[_max - 10:_max + 10].query('d<10').sort_values(by=['d']).index[0]
                except:
                    continue
                if abs(cycle.loc[_min].d) > abs(cycle.loc[_max].d):
                    cycle.loc[_min].d = cycle.loc[_min].d + cycle.loc[_max].d
                    cycle.loc[_max].d = 0
                    print('_min:{},_max:{}抵消'.format(_min, _max))
                elif abs(cycle.loc[_min].d) < abs(cycle.loc[_max].d):
                    cycle.loc[_max].d = cycle.loc[_min].d + cycle.loc[_max].d
                    cycle.loc[_min].d = 0
                    print('_min:{},_max:{}抵消'.format(_min, _max))

        for i in range(0, len(cycle)):
            # 一阶差分 往前5格滑窗+向后5格滑窗        limbs_all_dict[str(i)].to_excel('test_{}.xlsx'.format(i))
            if (abs(cycle[i:i + 5].d.sum()) < 10):  # and (i<5)
                stable.append(cycle.croods.iloc[i])
            # elif (abs(cycle[i:i+5].d.sum()) < 10) or (abs(cycle[i-4:i+1].d.sum()) < 10): #and (len(cycle[i:i+5])>2): #清除集合中只有一个点的情况（末尾点）
            #    stable.append(cycle.croods.iloc[i])
        return stable

    def count_touchdown_time(self):
        limbs_cycle_dict = {'cycle_name': [i for i in range(len(self.cyclelist))], 'lhoof_1': 0, 'lhoof_2': 0, 'lhoof_time': 0,
                            'rhoof_1': 0, 'rhoof_2': 0, 'rhoof_3': 0, \
                            'hind_lhoof_1': 0, 'hind_lhoof_2': 0, 'hind_lhoof_3': 0, 'hind_rhoof_1': 0, 'hind_rhoof_2': 0,
                            'hind_rhoof_3': 0}
        limbs_cycle_df = pd.DataFrame(limbs_cycle_dict)

        # for i in range(len(stable_range_list))
        for name, cyclelist in self.cycle_dict.items():

            print(name)

            if name == 'lhoof':
                for j in range(len(self.stable_range_list)):
                    stable = self.stable_range_list  #######################
                    limbs_cycle_df[name + '_1'][j] = str([stable[j][0], stable[j][1]])
                    print(str([stable[j][0], stable[j][1]]))
                continue
            for i in range(len(cyclelist)):

                stable = self._stable_time(cyclelist[i]) #######################
                print(stable)
                if len(stable) < 2:
                    print(stable)
                    continue

                stable_df = pd.DataFrame(stable)
                stable_df['d'] = stable_df[0] - stable_df[0].shift(1)
                d_point = stable_df.query('d>5')
                t = 3
                # for i in range(len(d_point)):
                if (len(d_point) > 1) and (d_point.index[-1] == stable_df.index[-1]):  # 将末尾单个的d_point点删除
                    d_point = d_point[:-1]
                if len(d_point) == 0:
                    if abs(stable[0] - stable[-1]) > t:
                        limbs_cycle_df[name + '_1'][i] = str([stable[0], stable[-1]])
                if len(d_point) == 1:
                    ###########range小于t的范围被认为是不完整的触地或者错误识别的触地
                    if (abs(stable[0] - stable_df.iloc[d_point.index[0] - 1].iloc[0]) < abs(
                            stable_df.iloc[d_point.index[0]].iloc[0] - stable[-1])) and (
                            abs(stable_df.iloc[d_point.index[0]].iloc[0] - stable[-1]) > t):
                        limbs_cycle_df[name + '_1'][i] = str([stable_df.iloc[d_point.index[0]].iloc[0], stable[-1]])
                        # continue
                    if (abs(stable[0] - stable_df.iloc[d_point.index[0] - 1].iloc[0]) > abs(
                            stable_df.iloc[d_point.index[0]].iloc[0] - stable[-1])) and (
                            abs(stable[0] - stable_df.iloc[d_point.index[0] - 1].iloc[0]) > t):
                        limbs_cycle_df[name + '_1'][i] = str([stable[0], stable_df.iloc[d_point.index[0] - 1].iloc[0]])
                        # continue
                    # limbs_cycle_df[name+'_1'][i] = str([stable[0],stable_df.iloc[d_point.index[0]-1].iloc[0]])
                    # limbs_cycle_df[name+'_2'][i] = str([stable_df.iloc[d_point.index[0]].iloc[0],stable[-1]])
                if len(d_point) == 2:
                    #############
                    range1 = abs(d_point.index[0] - stable_df.index[0])
                    range2 = abs(d_point.index[0] - d_point.index[-1])
                    range3 = abs(d_point.index[-1] - stable_df.index[-1])
                    if range1 == max(range1, range2, range3):
                        limbs_cycle_df[name + '_1'][i] = str([stable[0], stable_df.iloc[d_point.index[0] - 1].iloc[0]])
                    elif range2 == max(range1, range2, range3):
                        limbs_cycle_df[name + '_1'][i] = str([stable_df.iloc[d_point.index[0]].iloc[0],
                                                              stable_df.iloc[d_point.index[1] - 1].iloc[0]])
                    else:
                        limbs_cycle_df[name + '_1'][i] = str([stable_df.iloc[d_point.index[1]].iloc[0], stable[-1]])
                    # if min(range1,range2,range3) == range1:
                    #    d_point = d_point[1:]
                    # if min(range1,range2,range3) == range3:
                    #    d_point = d_point[:1]

                    # limbs_cycle_df[name+'_1'][i] = str([stable[0],stable_df.iloc[d_point.index[0]-1].iloc[0]])
                    # limbs_cycle_df[name+'_2'][i] = str([stable_df.iloc[d_point.index[0]].iloc[0],stable_df.iloc[d_point.index[1]-1].iloc[0]])
                    # limbs_cycle_df[name+'_3'][i] = str([stable_df.iloc[d_point.index[1]].iloc[0],stable[-1]])

        return limbs_cycle_df


def count_V(point, k):
    # k is a scale of the picture
    dt = (point.iloc[-5:]['croods'].mean() - point.iloc[:5]['croods'].mean()) / 180
    dx = (point.iloc[:5]['x'].mean() - point.iloc[-5:]['x'].mean()) / k
    v = dx / dt
    cycle_time = (point.iloc[-1]['croods'] - point.iloc[0]['croods']) / 180
    return v, cycle_time


def get_list(str1):
    t = re.findall(r'\[(.+?)\]', str1)
    t1 = t[0].split(',')
    b = [float(i) for i in t1]
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to the directory containing the pose data files')
    parser.add_argument('--scale', type=float, help='Scale of the video where 1 meter equals how many pixels')
    parser.add_argument('--out', type=str, help='Path of result output')
    args = parser.parse_args()
    scale = args.scale
    output_path = args.out
    pose_path = args.path
    pose_path_fir = file_name(pose_path)
    pose_path_fir.sort()
    
    #按照不同大小的缩放尺寸视频进行分类
    #path = 'F:/sheepvedios/玛曲跑步/主视频/模型/csv/all'

    # 创建类的实例化个体
    S = Sheep_individual()
    idic = creat_idic(pose_path_fir)

    # 前处理完成，开始后续分析
    limbs_all_dict = {}
    P = Cycle_count_peak()
    T = Cycle_count_touchtime()
    df_alldata = pd.DataFrame([])
    error = 0
    for i in range(len(idic)):

        p_class = idic[str(i)]
        id1 = str(i)
        # 刷新class
        # p_class.pretreatpoint(pose_path_fir[17])
        point = p_class.lhoof
        # 提取name
        name = pose_path_fir[int(id1)].split('test/')[-1].split('DLC')[0]

        # peak_n to count maxpoint
        maxpoint = P.peak_n(point, 1)
        # maxpoint_filt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        maxpoint = P.maxpoint_filt(point, maxpoint, ax1)

        # cycle_count
        '''
        ————————————————————————————————————————————————————————————
        用左前肢触地的一瞬间作为步态周期的开始，以连续的下一次左前肢触地为结束一个步态周期
        ————————————————————————————————————————————————————————————
        '''
        stable_start_array = T.stable_start(point)
        cycle_dict = T.create_dict(stable_start_array, ax2)  # self.cycle_dict
        fig.show()
        #fig.savefig(r'F:\项目数据\运动GWAS项目绘图\cycle\{}_frames-x.pdf'.format(name))

        limbs_cycle_df = T.count_touchdown_time()
        limbs_all_dict[str(i)] = limbs_cycle_df
        limbs_cycle_df = limbs_cycle_df.loc[:, (limbs_cycle_df != 0).any(axis=0)]  # 删除全部为0的那一列
        '''
        ————————————————————————————————————————————————————————————
        进行步态分类
        ————————————————————————————————————————————————————————————
        '''
        gait_df = pd.DataFrame({'cycle_name': [i for i in range(len(limbs_cycle_df))], 'Landing order': 0, 'V': 0,
                                'cycle_time': 0})
        raw = 0  # 用以计数遍历到达了第几行
        for cycle in limbs_cycle_df.itertuples():  # 完成字符串到列表的转换
            a_list = []
            for i in cycle[2:]:
                if i == 0:
                    a_list.append(0)
                    continue
                a_list.append(get_list(i)[0])
            a_df = pd.DataFrame(a_list)
            a_df = a_df.sort_values(by=[0])[a_df[0] > 0]  # 对周期的开始节点进行排序
            r = ''
            for order in a_df.T:
                r = r + str(order + 1)
            r = r.ljust(4, 'X')  # 将为0或者计算错误的值补齐
            print(r)
            gait_df['Landing order'].iloc[raw] = r
            raw += 1
        '''
        ————————————————————————————————————————————————————————————
        以步态周期为单位计算奔跑速度***
        ————————————————————————————————————————————————————————————
        '''
        eye = p_class.bodypart2
        seppoint = T.stable_start(point)
        eye_cycle_list = T._cycle_count(eye, seppoint)
        if len(eye_cycle_list[-1]) == 0:
            eye_cycle_list = eye_cycle_list[:-1]
        V_list = []
        for j in eye_cycle_list:
            if len(j) == 0:
                continue
            name = pose_path_fir[int(id1)].split('all/')[-1]  # Gallop/
            V, cycle_time = count_V(j, scale)
            print(V, cycle_time)
            V_list.append([V, cycle_time])

        for i in range(len(V_list)):
            gait_df.loc[i, 'V'] = float(V_list[i][0])
            gait_df.loc[i, 'cycle_time'] = float(V_list[i][1])
        merge_df = pd.concat([limbs_cycle_df, gait_df.iloc[:, 1:]], axis=1)
        merge_df_head = pd.concat([pd.DataFrame({'cycle_name': [name.split('DLC')[0]]}), merge_df], ignore_index=True)
        df_alldata = pd.concat([df_alldata, merge_df_head], ignore_index=True)

    # excel的格式化输出
    excel = pd.ExcelWriter('MQ_T_all_V.xlsx', engine='xlsxwriter')

    df_alldata.to_excel(excel, 'aa', index=None)
    sheet = excel.sheets['aa']
    sheet.set_column(1, 10, 12)
    sheet.set_column(0, 0, 12)
    excel.close()

    df_alldata = df_alldata.reset_index(drop=True)
    df_alldata_str = df_alldata.astype("str")
    # 步态周期过滤
    df_alldata_str_cycle01_V1to10 = df_alldata_str.query('"0.5">cycle_time>"0.1" or V=="-"').query(
        '("9">V>"1") or (V =="-")')

    df_alldata_str = df_alldata_str.where(df_alldata.notnull(), '-')
    df_alldata_cycle01 = df_alldata_str.query('("0.5">cycle_time>"0.1") or (cycle_time=="-")')
    df_alldata_cycle01_V1 = df_alldata_cycle01.query('V>"2" or V =="-"')
    df_alldata_cycle01_V1 = df_alldata_cycle01_V1.reset_index().drop('index', axis=1)  # 重置行索引,提取个体分割的索引值，提取多个周期中最大的三个速度值
    # df_alldata_str_cycle01_V1to10 = df_alldata_str_cycle01_V1to10.
    index_list = df_alldata_cycle01_V1.query('cycle_time == "-"').index
    for i in range(len(index_list) - 1):
        s = index_list[i]
        t = index_list[i + 1]
        V_max = df_alldata_cycle01_V1[s:t].query('V!="-"').V.astype('float').max()  # 提取对应区段，找到速度最大的三个值，求他们的平均值
        # print(df_alldata_str_cycle01_V1to10[s:s+1].cycle_name.iloc[0]+'=' +str(V_max))
        df_alldata_cycle01_V1.iloc[s, 6] = V_max

    df_alldata_cycle01_V1.query('cycle_time!="-"')[['cycle_name', 'V']].to_excel('MQ_T_V_0610.xlsx')
    V_max_df = df_alldata_cycle01_V1.query('cycle_time=="-"')[['cycle_name', 'V']]
    V_max_df.to_excel(output_path+'\\'+'MQ_T_name_V.xlsx')
