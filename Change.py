#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:52:07 2021

@author: hjh
"""

import numpy as np
import matplotlib.pyplot as plt

class Change():
    '''
    传入的是过滤后的new序列
    '''
    def __init__(self, seq1=[], seq2=[], indi=2):
        self.seq1 = np.array(seq1)
        self.seq2 = np.array(seq2)
        self.indi = indi
        print('个体数indi=',indi)
        pass
    def test(self, seq):
        
        if len(seq)==0:
            print('空集合')
            return seq
        seq = np.array(seq)
        seq_a = np.array(seq)[:-1] #切片创建前后差一个位点的两个矩阵
        seq_b = np.array(seq)[1:] 
        deb = seq_b[:,1:3] - seq_a[:,1:3] #先对矩阵中的内容进行筛选，选出横纵坐标，然后进行矩阵相减得到d
        self.ED = np.sqrt((deb ** 2).sum(axis=1)) #计算欧氏距离
        seq_ED = np.c_[seq_b[:],self.ED] #拼接矩阵，在点集中增加欧氏距离为最后一列（此处的欧式距离为该点与之上一个点之间的距离，所以删除了第一个点）
        return seq_ED
    
    def cut(self, seq_ED, threshold=300):
        '''
        threshold在这里指前后两点距离超过该值则判断为交换点，进行切割
        fragment会返回一个列表，其中储存了分割的各段序列
        '''
        if len(seq_ED) == 0 :
            return seq_ED
        self.threshold = threshold
        pos = np.where(seq_ED[:,-1] > self.threshold)[0] #断点检测,用where找到欧氏距离大于阈值的位置
        if len(pos) > 0:    
            fragment = np.split(seq_ED, pos)
            return fragment
        else:
            return [seq_ED]
    
    def count_ED(self, np1, np2):
        '''
        距离计算
        '''
        d = np1[1:3] - np2[1:3]
        ED = np.sqrt((d ** 2).sum(axis=0))
        return ED
    def plt(self, seq, labelx = 'x', labely = 'y'):
        x = seq[:,1]
        y = seq[:,2]
        plt.xlabel(labelx,fontsize=13)
        plt.ylabel(labely,fontsize=13)
        plt.scatter(x,y)
        plt.show

    def tx(self, new_seq1):
        
        #对比被取出序列与所有其他序列的匹配程度
        s_count = np.array([])
        e_count = np.array([])
        for i in range(len(self.fragment)):    
            targetfrag = self.fragment[i] 
            startpoint_ED = self.count_ED(new_seq1[0],targetfrag[-1]) #计算开始点和结束点与指定片段的前后连接处距离
            endpoint_ED = self.count_ED(new_seq1[-1], targetfrag[0])
            #print(startpoint_ED,endpoint_ED)
            s_count = np.append(s_count,startpoint_ED)
            e_count = np.append(e_count,endpoint_ED)
        #print(s_count,e_count)
        
        if (len(s_count) == 0) or (min(np.min(s_count), np.min(e_count)) > self.threshold) :
            print('无匹配段,结束拼接')
            print(len(new_seq1))
            self.new_seq2 = new_seq1
            #self.plt(self.new_seq2, labelx='frames(correct1)', labely= 'h')
            return new_seq1 
        if np.min(s_count) < np.min(e_count):
            tar_idex = np.where(s_count==np.min(s_count))[0][0] #找到s_count中的最小ED值对应序号
            #if len(self.fragment.pop(tar_idex)) >= 15: #设置只有长度大于15的片段才能被拼接
            new_seq1_s = np.r_[self.fragment.pop(tar_idex), new_seq1] #进行拼接
            print('已拼接到基础序列前')
            #else:
            #   new_seq1_s = new_seq1
            self.tx(new_seq1_s)  #进行递归
        else:
            tar_idex = np.where(e_count==np.min(e_count))[0][0]
            print(tar_idex,len(self.fragment))
            new_seq1_e = np.r_[new_seq1, self.fragment.pop(tar_idex)]
            print('已拼接到基础序列后')
            self.tx(new_seq1_e)
    def combin(self):
        
        seq1_ED = self.test(self.seq1)
        fragment1 = self.cut(seq1_ED)
        seq2_ED = self.test(self.seq2)
        fragment2 = self.cut(seq2_ED)
        if len(fragment1) == len(fragment2) == 1:
            print('indi=',self.indi)
            if self.indi == 2:
                print('未发生标签交换')
                self.new_seq1 = fragment1[0]
                self.new_seq2 = fragment2[0]
                t = 'over'
                return t
            else:
                print("单动物两轨迹合并")
                self.fragment = fragment1 + fragment2
        elif (len(fragment1) == 0) and (len(fragment2) != 0):
            self.fragment = fragment2
        elif (len(fragment1) != 0) and (len(fragment2) == 0):
            self.fragment = fragment1
        else:
            self.fragment = fragment1 + fragment2 #先合并两个列表
        '''
        思路：用贪心算法做局部最优的选择，最优化就是指进行拼接的时候要求接头处的欧氏距离最小（或者在一定范围内）
        '''
        
        if len(self.fragment) == 1:
            self.new_seq1 = self.fragment.pop(0)
            self.new_seq2 = np.array([])
            print('只有一条序列')
            return '只有一条序列'
        elif self.indi == 1:
            print("单动物")
            
        self.fragment = sorted(self.fragment, key=lambda l:-len(l)) #对fragment列表按序列长度从大到小进行排序
        #print(self.fragment)
        self.fragment_1 = []
        for i in self.fragment:
            if len(i) != 0:
                self.fragment_1.append(i)
            print('frag中的序列长度：',len(i))
        self.fragment = self.fragment_1
        self.new_seq2 = self.fragment.pop(0)  #从列表中取出最长的序列,先取2，得到结果后把标签贴为1

        print('frag:',len(self.fragment)) #查看当前fragment中片段数
        self.tx(self.new_seq2)
        self.new_seq1 = self.new_seq2
        if (len(self.fragment) == 1) : #如果fragment中只剩下一个序列，且indi=2，那么将它作为seq2，如果无序列，赋为0
            if self.indi == 2:
                self.new_seq2 = self.fragment[0]
            #if indi == 1:
                
        elif len(self.fragment) == 0:
            self.new_seq2 = 0
        else:
            self.new_seq2 = self.fragment.pop(0)
            self.tx(self.new_seq2)
            #self.plt(self.new_seq2, labelx='frames(correct2)', labely= 'h')