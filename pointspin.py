#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:38:02 2021

@author: hjh
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import filt_pretreatpoint_tendst as fpt

'''
逆时针3.1
顺时针1.5（实际为2.5-1，其中2.5是偏角，1是正常坡度1°）
'''
# 绕pointx,pointy逆时针旋转
def Nrotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return nRotatex, nRotatey
# 绕pointx,pointy顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return sRotatex,sRotatey

def pt(new1,new2):
    for i in [new1, new2]:
            
        np_new = np.array(i)
        x = np_new[:,1]
        y = np_new[:,2]
    
        plt.scatter(x,y)
        plt.show()
        
def spin_l(new,angle,plot=False):
    new = np.array(new)
    if plot:    
        plt.scatter(new[:,1],new[:,2])
        plt.show()
    new[:,2]=1080-new[:,2]
    n = new[:,1:3]
    spin1_x = []
    spin1_y = []
    for pointx,pointy in n:
        sPointx ,sPointy = Nrotate(math.radians(angle),pointx,pointy,980,540)
        spin1_x.append(sPointx)
        spin1_y.append(sPointy)
    new[:,1] = spin1_x
    new[:,2] = spin1_y
    if plot:
        plt.scatter(spin1_x,spin1_y)
        plt.show()
    return new
    
def spin_r(new,angle,plot=False):
    new = np.array(new)
    if plot:
        plt.scatter(new[:,1],new[:,2])
        plt.show()
    new[:,2]=1080-new[:,2]
    n = new[:,1:3]
    spin1_x = []
    spin1_y = []
    for pointx,pointy in n:
        sPointx ,sPointy = Srotate(math.radians(angle),pointx,pointy,980,540)
        spin1_x.append(sPointx)
        spin1_y.append(sPointy)
    new[:,1] = spin1_x
    new[:,2] = spin1_y  
    if plot:
        plt.scatter(spin1_x,spin1_y)
        plt.show()
    return new
                    
'''
pointx = 0
pointy = 10
sPointx ,sPointy = Nrotate(math.radians(5),pointx,pointy,980,540)
print(sPointx,sPointy)
plt.plot([0,pointx],[0,pointy])
plt.plot([0,sPointx],[0,sPointy])

plt.xlim(-10.,10.)
plt.ylim(-10.,10.)
plt.xticks(np.arange(-3.,3.,1))
plt.yticks(np.arange(-3.,3.,1))
plt.show()
'''


if __name__ == '__main__':
    
    path = '/home/hjh/DLC1/output/jump/MQ_jump/MQ_jump_6/T127_0670_128_2830_1DLC_resnet50_MQ_jump_6MApr16shuffle1_50000_bx_filtered.csv'
    new1= fpt.pretreatpoint(path,individual=1)
    new2= fpt.pretreatpoint(path,individual=2)
    
    new1 = np.array(new1)
    new2 = np.array(new2)
        
    new1[:,2]=1080-new1[:,2]
    new2[:,2]=1080-new2[:,2]
    
    n1 = new1[:,1:3]
    n2 = new2[:,1:3]
    pt(new1,new2)
    
    spin1 = np.array([])
    spin2 = np.array([])
    spin1_x = []
    spin1_y = []
    for pointx,pointy in n1:
        sPointx ,sPointy = Nrotate(math.radians(-2.),pointx,pointy,980,540)
        spin1_x.append(sPointx)
        spin1_y.append(sPointy)
        continue
    new1[:,1] = spin1_x
    new1[:,2] = spin1_y
    plt.scatter(spin1_x,spin1_y)
    
    '''
    new = spin_r(new1,-1.5,individual=1)
    plt.scatter(new[:,1],new[:,2])
    plt.show()
'''