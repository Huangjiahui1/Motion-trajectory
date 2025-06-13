# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:16:32 2023

@author: h
"""

import pandas as pd
import os
import numpy as np
path = r'F:\sheepvedios\output\run\MQ_run_all_fix_314'

path = r'F:\sheepvedios\output\run\allcsv'


def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            
            allname = file_dir + '\\' + file   #it's a difference between linux and win
            L.append(allname)
    return L

L = file_name(path)
test = pd.read_csv(L[0],header = 2)

for i in range(len(L)):
    if i == 0:    
        exc = pd.read_csv(L[0],header = 2)
    else:
        exc = exc.append(pd.read_csv(L[i],header = 2))
        
exc.columns = ['croods', 'bodypart1_x', 'bodypart1_y', 'bodypart1_lh', 'bodypart2_x','bodypart2_y', 'bodypart2_lh', \
              'bodypart3_x', 'bodypart3_y', 'bodypart3_lh','b4_x', 'b4_y', 'b4_lh', 'b5_x', 'b5_y', 'b5_lh', \
                  'b6_x', 'b6_y', 'b6_lh', 'b7_x', 'b7_y', 'b7_lh', 'b8_x', 'b8_y', 'b8_lh','lhoof_x', 'lhoof_y', 'lhoof_lh',\
                      'rhoof_x', 'rhoof_y','rhoof_lh', 'hind_lhoof_x', 'hind_lhoof_y', 'hind_lhoof_lh', 'hind_rhoof_x', 'hind_rhoof_y', 'hind_rhoof_lh']

exc = exc.query('bodypart1_lh>0.2 and bodypart2_lh>0.2 and bodypart3_lh>0.2 and b4_lh>0.2')
export = exc[['bodypart1_lh','bodypart2_lh','bodypart3_lh','b4_lh','b5_lh','b6_lh','b7_lh','b8_lh','lhoof_lh','rhoof_lh','hind_lhoof_lh','hind_rhoof_lh']]
np.average(export,axis=0) #用于计算每个点的总平均置信度
df_hotmap = pd.DataFrame(np.arange(120).reshape(10,12))
df_hotmap.columns = ['bodypart1_lh','bodypart2_lh','bodypart3_lh','b4_lh','b5_lh','b6_lh','b7_lh','b8_lh','lhoof_lh','rhoof_lh','hind_lhoof_lh','hind_rhoof_lh']

def count_frq(df,name):
    for i in range(0,10):
        df_hotmap['name'] = df.query('0.1*@i<lh<0.1*(@i+1)')

all_average_lh = []
for i in range(len(export.columns)):
    name = export.columns[i]
    if i == 0:    
        DF_lh = pd.DataFrame(export.iloc[:,i])
        DF_lh['name'] = name
        DF_lh.columns = ['lh','name']
        for j in range(0,10):
            df_hotmap.iloc[j,i] = len(DF_lh.query('0.1*@j<lh<0.1*(@j+1)'))
        all_average_lh.append(np.average(DF_lh.lh))
        
    else:
        add_DF_lh = pd.DataFrame(export.iloc[:,i])
        add_DF_lh['name'] = name
        add_DF_lh.columns = ['lh','name']
        # add_DF_lh = add_DF_lh.query('lh>0.2')
        DF_lh = DF_lh.append(add_DF_lh, ignore_index=True)
        for j in range(0,10):
            df_hotmap.iloc[j,i] = len(add_DF_lh.query('0.1*@j<lh<0.1*(@j+1)'))
        all_average_lh.append(np.average(add_DF_lh.lh))
            
sum1 = df_hotmap.sum()
df_hot_map_rate = df_hotmap/df_hotmap.sum()
rev = df_hot_map_rate.iloc[::-1]
rev.to_csv(r'F:\项目数据\运动GWAS项目绘图\附图\likelihood_allrun_1081_hotmap_rate_rev.csv', sep=' ', index=None)    
DF_lh['num'] = range(1704575)   
DF_lh.to_csv(r'C:\Users\h\Desktop\all_point_lh.txt', sep=' ', index=None)        
'''
________________________________________________________________________________________________________________________________________________
#jump
'''
path_jump = r'F:\softln\DLC1\output\jump_2.2C\all'

L = file_name(path_jump)
test = pd.read_csv(L[0],header = 2)
test.iloc[:,1:].dropna(how='all')
col = ['croods', 'bodypart1_x', 'bodypart1_y', 'bodypart1_lh', 'bodypart2_x','bodypart2_y', 'bodypart2_lh', \
              'bodypart3_x', 'bodypart3_y', 'bodypart3_lh','b4_x', 'b4_y', 'b4_lh', 'bodypart2_1_x', 'bodypart2_1_y', 'bodypart2_1_lh', \
              'bodypart2_2_x','bodypart2_2_y', 'bodypart2_2_lh', \
              'bodypart2_3_x', 'bodypart2_3_y', 'bodypart2_3_lh','b2_4_x', 'b2_4_y', 'b2_4_lh',]
for i in range(len(L)):
    if i == 0:    
        exc = pd.read_csv(L[0],header = 3)
        exc.columns = col
    else:
        x = pd.read_csv(L[i],header = 3)
        if len(x.columns) < 20:
            continue
        x.columns = col
        exc = exc.append(x)

df_hotmap = pd.DataFrame(np.arange(80).reshape(10,8))
export_lh = exc[['bodypart1_lh','bodypart2_lh','bodypart3_lh','b4_lh','bodypart2_1_lh','bodypart2_2_lh','bodypart2_3_lh','b2_4_lh']]
export_1_lh = exc[['bodypart1_lh','bodypart2_lh','bodypart3_lh','b4_lh']]
export_2_lh = exc[['bodypart2_1_lh','bodypart2_2_lh','bodypart2_3_lh','b2_4_lh']]
export_1 = export_1_lh.dropna(subset=['bodypart1_lh'])
export_2 = export_2_lh.dropna(subset=['bodypart2_1_lh'])

export_1 = export_1.reset_index().iloc[:,1:]
export_2 = export_2.reset_index().iloc[:,1:]
export_lh = pd.concat([export_1,export_2],axis=1)
df_hotmap.columns = ['bodypart1_lh','bodypart2_lh','bodypart3_lh','b4_lh','bodypart2_1_lh','bodypart2_2_lh','bodypart2_3_lh','b2_4_lh']

all_average_lh = []
for i in range(len(export_lh.columns)):
    name = export_lh.columns[i]
    if i == 0:    
        DF_lh = pd.DataFrame(export_lh.iloc[:,i])
        DF_lh['name'] = name
        DF_lh.columns = ['lh','name']
            DF_lh = DF_lh.query('lh>0.1')
        for j in range(0,10):
            df_hotmap.iloc[j,i] = len(DF_lh.query('0.1*@j<lh<=0.1*(@j+1)'))
        DF_lh = DF_lh.dropna()
        all_average_lh.append(np.average(DF_lh.lh))
        
    else:
        add_DF_lh = pd.DataFrame(export_lh.iloc[:,i])
        add_DF_lh['name'] = name
        add_DF_lh.columns = ['lh','name']
        add_DF_lh = add_DF_lh.query('lh>0.1')
        DF_lh = DF_lh.append(add_DF_lh, ignore_index=True)
        DF_lh = DF_lh.dropna()
        for j in range(0,10):
            df_hotmap.iloc[j,i] = len(add_DF_lh.query('0.1*@j<lh<=0.1*(@j+1)'))
        all_average_lh.append(np.average(add_DF_lh.lh))


sum1 = df_hotmap.sum()
df_hot_map_rate = df_hotmap/df_hotmap.sum()

df_hot_map_rate.to_csv(r'C:\Users\h\Desktop\lh_hot_map_rate10.csv',index=None,sep=' ')
rev = df_hot_map_rate.iloc[::-1].iloc[:-1,:]
rev.to_csv(r'C:\Users\h\Desktop\lh_hotmap_jump_rate10_rev.csv', sep=' ', index=None)  
DF_lh.to_csv(r'C:\Users\h\Desktop\all_point_jump_lh.txt', sep=' ', index=None)     


###################画图
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 假设你的数据存储在一个DataFrame中，名为rev



# 定义自定义颜色映射
colors = [
    (0.0, '#426AA0'),  # 0.0
    (0.2, '#A1C4D6'),  # 0.2
    (0.4, '#EAF2D5'),  # 0.4
    (0.6, '#F2DD97'),  # 0.6
    (0.8, '#E58456'),  # 0.8
    (1.0, '#C13328')   # 1.0
]

# 创建自定义颜色映射
cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

# 绘制热图
plt.figure(figsize=(12, 8))
sns.heatmap(rev, cmap=cmap, annot=False, fmt='g', linecolor='white', linewidths=0.5)
plt.xlabel('Body Part')
plt.ylabel('Frame Number')
plt.title('Likelihood Heat Map for All Body Parts')

# 保存为PDF
plt.savefig(r'F:\项目数据\运动GWAS项目绘图\附图\allrun1081_likelihood_仅point2过滤lh0.2.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()




log = pd.read_csv(r'F:\softln\DLC2.2C\model_running20220302\MQ_runing_merge-M1-posepres-2022-03-02\dlc-models\iteration-0\MQ_runing_merge-M1Mar2-trainset95shuffle3\train\log_1.txt',sep='\t')

log_1000.to_csv('F:\项目数据\运动GWAS项目绘图\主图\log_runmodel_1000.txt',index=None,sep='\t')












