import torch
import torch.optim as optim
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
#  preprocessing预处理模块
from sklearn import preprocessing


features = pd.read_csv('temps.csv')

# 默认前五行
print('head: ', features.head())
print('数据维度: ', features.shape)


# 年月日
years = features['year']
months = features['month']
days = features['day']

# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

print(dates[:5])
# 独热编码 自动判断字符串进行编码
features = pd.get_dummies(features)
print(features.head())

# 标签
labels = np.array(features['actual'])
# 特征中去掉标签
features = features.drop('actual', axis=1)
# 名字单独保存一下
feature_list = list(features.columns)
# 转换成合适的格式
features = np.array(features)

print(features.shape)



# 标准化
input_features = preprocessing.StandardScaler().fit_transform(features)
print(input_features[0])

# 构建网络模型
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)

# 权重参数初始化
weights = torch.randn((14, 128), dtype=float, requires_grad=True)
biases = torch.randn(128, dtype=float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
biases2 = torch.randn(1, dtype=float, requires_grad=True)

learning_rate = 0.001
losses = []

for i in range(1000):
    # 计算隐层
    hidden = x.mm(weights) + biases
    # 加入激活函数 ?
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 通计算损失
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())

    # 打印损失值
    if i % 100 == 0:
        print("loss:", loss)
    # 反向传播计算
    loss.backward()

    # 更新参数
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)
    # 每次迭代清空操作
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
