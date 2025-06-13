# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:56:34 2025

@author: h
"""
import pandas as pd
GRID2_snp = pd.read_csv(r'F:\data\sheep227_GRID2_31647731-32943000.recode2.vcf',sep='\t')
GRID2_snp = GRID2_snp.replace('0|1','0/1').replace('1|0','0/1').replace('0|0','0/0').replace('1|1','1/1')
data_all = GRID2_snp.iloc[:,9:].T
data_all['name'] = data_all.index
data_all['name'] = data_all['name'].astype(str)
agali = data_all[data_all.name.str.contains('ARG')]
admix = pd.read_excel(r'F:\项目数据\DLC-GWAS文章整理\提交版本\20250509\Supplementary Tables.xlsx',header=1,sheet_name="Supplementary Table 9")
admix.columns = ['name','b','c','type']
admix[['name', 'type']]
admix.name = admix.name.astype(str)

data_all = pd.merge(data_all, admix, on='name')

#ZY = data_all.query('type=="ZY"')

# 去掉最后一列“name”
data = agali.iloc[:, :-1]

# 找到所有值都相同的列
columns_all_same = data.apply(lambda col: col.nunique() == 1, axis=0)

# 统计所有值都相同的列数
num_columns_all_same = columns_all_same.sum()

print(f"所有值都相同的列数: {num_columns_all_same}")

# 获取这些列的列名
columns_all_same_names = data.columns[columns_all_same].tolist()
print("所有值都相同的列名: ", columns_all_same_names)

agali = agali[columns_all_same_names]
agali = agali.loc[:, ~(agali == '0/1').any()]
columns_all_same_names = agali.columns.tolist()
#选取AGALI个体的纯合位点计算血统
data_pure = data_all[columns_all_same_names]
agali_ref = pd.DataFrame(agali[columns_all_same_names].iloc[0,:])
data_pure['name'] = data_all['name']

# 从 data_pure 中提取不含 "ARG" 字符的行
filtered_data_pure = data_pure[~data_pure['name'].str.contains('ARG')].iloc[:,:-1]
filtered_data_pure.columns = [i for i in range(10980)]
agali_ref = agali_ref.reset_index().iloc[:,1:]
# 初始化一个列表来存储相同的比例
proportion_same = []

# 遍历每一行并计算相同的比例
for index, row in filtered_data_pure.iterrows():
    same_count = 0
    total_count = 0
    for i, value in enumerate(row):
        if value == './.' or (agali_ref.iloc[i] == './.')[0]:
            continue
        if (value == '0/0') and (agali_ref.iloc[i][0] == '0/0'):
            continue
        total_count += 1
        if (value == '1/1') and (agali_ref.iloc[i][0] == "1/1"):
            same_count += 1
        elif value == '0/1'  and (agali_ref.iloc[i][0] == "1/1") :
            same_count += 0.5
    proportion = same_count / total_count if total_count > 0 else 0
    proportion_same.append(proportion)

# 将相同的比例添加到 data_pure 的最后一列
filtered_data_pure['proportion_same'] = proportion_same
filtered_data_pure['name'] = data_pure['name']
filtered_data_pure['type'] = data_all['type']
# 合并回原始数据
data_pure = data_pure.merge(filtered_data_pure[['name', 'proportion_same']], on='name', how='left')

print(data_pure)

phenotype = pd.read_excel(r'F:\项目数据\DLC-GWAS文章整理\提交版本\20250509\Supplementary Tables.xlsx',header=1,sheet_name="Supplementary Table 5")
phenotype.columns = ['name','type',',speed','height']
phenotype.name = phenotype.name.astype(str)
pd.merge(filtered_data_pure, phenotype, on = 'name').iloc[:,-6:].to_excel(r'F:\data\all_pheno_proportion0520.xlsx')
data_phenotype = pd.merge(filtered_data_pure, phenotype, on = 'name').iloc[:,-5:]

cov = pd.read_excel(r'F:\项目数据\DLC-GWAS文章整理\提交版本\20250509\Supplementary Tables.xlsx',header=1,sheet_name="Supplementary Table 6")
cov = cov[['Sample ID', 'Sex','Weight (kg)','Age (y)']]
cov.columns = ['name','sex','weight','age']
cov.name = cov.name.astype(str)
data_phenotype.name = data_phenotype.name.astype(str)
all_data = pd.merge(data_phenotype, cov, on = 'name')
all_data.to_excel(r'F:\data\all_pheno_cov_proportion_0520.xlsx')