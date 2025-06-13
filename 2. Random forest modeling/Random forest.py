# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:31:10 2025

@author: h
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


hybrid_random_forst = pd.read_csv(r'F:\data\杂交羊随机森林input.txt',sep='\t')
tibet_random_forst = pd.read_csv(r'F:\data\藏羊随机森林input.txt',sep='\t')
data = hybrid_random_forst
data = tibet_random_forst
# 定义特征和目标变量
#先算奔跑
data = data.iloc[data.V.dropna().index]
data = data.iloc[data.H.dropna().index]
X = data.drop(['V','Age','H','Unnamed: 10'], axis=1)#
y = data['V']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用网格搜索寻找最佳树的数目
param_grid = {
    'n_estimators': [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"Best number of trees: {grid_search.best_params_['n_estimators']}")

# 使用最佳参数训练模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 绘制特征重要性
importances = best_rf.feature_importances_
features = X.columns
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest')
plt.gca().invert_yaxis()
#plt.savefig(r'F:\项目数据\运动GWAS项目绘图\主图\Figure2_随机森林Feature Importances.pdf')
plt.savefig(r'F:\项目数据\运动GWAS项目绘图\主图\Figure2_随机森林_Tibetanjump_Feature Importances.pdf')
plt.show()

# 输出特征重要性
print(feature_importances)

# 可视化真实值与预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# 可视化不同树的数目的均方误差得分
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_estimators'], -results['mean_test_score'], marker='o', linestyle='dashed', color='r')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Number of Trees')
#plt.savefig(r'F:\项目数据\运动GWAS项目绘图\主图\Figure2_随机森林不同tree数目均方误差.pdf')
plt.savefig(r'F:\项目数据\运动GWAS项目绘图\主图\Figure2_随机森林_Tibetanjump_不同tree数目均方误差.pdf')
plt.show()