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

# print(dates[:5])
# 独热编码 自动判断字符串进行编码
features = pd.get_dummies(features)
# print(features.head())

# 标签
labels = np.array(features['actual'])
# 特征中去掉标签
features = features.drop('actual', axis=1)
# 名字单独保存一下
feature_list = list(features.columns)
# 转换成合适的格式
features = np.array(features)

# print(features.shape)


# 标准化
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features[0])

# 构建网络模型
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)

# 简化操作
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
#
my_nn = torch.nn.Sequential(
    # 输入样本数量
    torch.nn.Linear(input_size, hidden_size),
    # 激活函数
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)
# MSEloss
cost = torch.nn.MSELoss(reduction='mean')
# 优化器  adam动态调整学习率
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)
# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch 方法进行训练
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 预测训练结果

x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()

# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()

# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
