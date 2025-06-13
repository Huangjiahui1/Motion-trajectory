# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:59:22 2025

@author: h
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
vcf_phased = pd.read_csv(r'F:\data\sheep227_GRID2_32108974-32125813.beagle.vcf',sep='\t',header=8)

vcf_data = vcf_phased.iloc[:,9:].T
vcf_data['name'] = vcf_data.index
vcf_data['name'] = vcf_data['name'].astype(str)
H = pd.read_csv(r'F:\data\H_214_20240621.txt',sep='\t')
merge = pd.merge(vcf_data, H, on='name', how = 'left')
df1 = merge.iloc[:,:-2].apply(lambda col: col.str[0])
df2 = merge.iloc[:,:-2].apply(lambda col: col.str[2])

df1['name'] = merge['name']
df2['name'] = merge['name']
df1['H'] = merge['H']
df2['H'] = merge['H']

all_merge = pd.concat([df1,df2])
#all_merge.iloc[:,:-2] = all_merge.iloc[:,:-2].astype(int)
pca = PCA(n_components=2)
hap_pca = pca.fit_transform(all_merge.iloc[:,:-2])
plt.scatter(hap_pca[:, 0], hap_pca[:, 1])
kmeans = KMeans(n_clusters=4, random_state=0).fit(hap_pca)
#PCA绘图
plt.scatter(hap_pca[:, 0], hap_pca[:, 1], c=kmeans.labels_)
plt.scatter(hap_pca[:, 0], hap_pca[:, 1], c=all_merge['labels'])
plt.savefig(r'F:\data\pca.pdf')

all_merge['labels'] = kmeans.labels_
all_merge.to_excel(r'F:\data\hap_454_merge.xlsx',index=None)
admixture = pd.read_csv(r'F:\data\admixture.txt',sep='\t')
all_merge = pd.merge(all_merge, admixture, on='name')
all_merge.to_excel(r'F:\data\hap_454_merge_admix.xlsx',index=None)
all_merge = pd.read_excel(r'F:\data\hap_454_merge_admix.xlsx')
#热图
plot=sns.heatmap(all_merge.iloc[:,:-5].astype('int'))
plt.savefig(r'F:\data\hap_454_heatmap.pdf')

#单倍型组合表型绘图
hap_454 = pd.read_excel(r'F:\data\hap_454_merge.xlsx',sheet_name='Sheet2')
x = hap_454[['name','labels','H_R']]

# 定义一个函数来处理 labels 列的合并
def merge_labels(labels):
    # 将 labels 转换为字符串列表
    labels_str = [str(label) for label in labels]
    # 对列表进行排序，确保 0_1 和 1_0 等同
    labels_str.sort()
    # 用 _ 连接
    return '_'.join(labels_str)

# 定义一个函数来检查 H_R 列的值是否一致
def check_hr(hr_values):
    unique_hr = set(hr_values)
    if len(unique_hr) > 1:
        raise ValueError(f"H_R values are not consistent: {unique_hr}")
    return unique_hr.pop()

# 按照 name 列分组，并应用自定义的聚合函数
grouped = x.groupby('name').agg({
    'labels': merge_labels,
    'H_R': check_hr
}).reset_index()

print(grouped)
grouped.to_excel(r'F:\data\单倍型组合.xlsx')