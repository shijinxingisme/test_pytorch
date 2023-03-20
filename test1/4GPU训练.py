import numpy as np
import torch.nn as nn
import torch

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
print(x_train.shape)

y_values = [2 * i + 1 for i in range(11)]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_train.shape)


# 模型

class LinearRegressionModel(nn.Module):
    def __init__(self, a, b):
        super(LinearRegressionModel, self).__init__()
        # 全连接层
        self.linear = nn.Linear(a, b)

    # 前向传播
    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


# CUDA CPU 校验
# print("cuda have ? ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("model : ", model)
criterion = nn.MSELoss()

epochs = 1000

learning_rate = 0.01
# 优化器 SGD  优化参数  lr 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 损失函数
criterion = nn.MSELoss()


for epoch in range(epochs):
    epoch+=1
    #数据转化
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    #梯度清零每一次迭代
    optimizer.zero_grad()
    #前向传播
    outputs = model(inputs)
    #计算损失
    loss = criterion(outputs,labels)
    #反向传播
    loss.backward()
    #更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch,loss.item()))
