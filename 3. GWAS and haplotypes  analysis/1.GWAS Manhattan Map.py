# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:50:18 2025

@author: h
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# sample data
df = pd.read_csv('C:\\Users\\h\\Desktop\\manhat_R_input.csv',sep='\t')
#running speed
path = r'F:\download\run_avg3\result_sex_env_pca3.assoc.txt'
#jumping height
path = r'E:\sheep214_weight_sex_pca.assoc.txt'

name1 = path.split('\\')[-1].split('.assoc.txt')[0]
df = pd.read_csv(path,sep='\t')

df = df[['chr','rs','ps','p_wald']]

df.columns = ['chromosome','rs','ps','pvalue'] 
df.chromosome = df.chromosome.astype(str)
# -log_10(pvalue)
df['minuslog10pvalue'] = -np.log10(df.pvalue)
df.chromosome = df.chromosome.astype('category')
df.chromosome = df.chromosome.cat.set_categories([str(i) for i in range(1,28)], ordered=True)
# How to plot gene vs. -log10(pvalue) and colour it by chromosome?
df['ind'] = range(len(df))
df_grouped = df.groupby(('chromosome'))
 
# manhattan plot
fig = plt.figure(figsize=(10,4),dpi=100) 
ax = fig.add_subplot(111)
 
colors = ["#C0B4B1","#285C5C"]#,"#EFDC05","#E53A40","#090707"
x_labels = []
x_labels_pos = []
for num, (name, group) in enumerate(df_grouped):
    group.plot(kind='scatter', x='ind', y='minuslog10pvalue',color=colors[num % len(colors)], ax=ax,)
    x_labels.append(name)
    x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
# add grid
#ax.grid(axis="y",linestyle=None,linewidth=.5,color="gray")
ax.tick_params(direction='out',labelsize=13)
ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels)
#不显示上和右侧边框
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_xlim([0, len(df)])
ax.set_ylim([0, int(df.minuslog10pvalue.max())+1])
# x axis label
ax.set_xlabel('Chromosome',size=20)
plt.savefig('F:\项目数据\运动GWAS项目绘图\附图\Manhattan Plot {}.png'.format(name1),dpi=900,bbox_inches='tight',facecolor='white')
plt.show()
