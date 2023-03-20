import torch

print(torch.__version__)

#返回填充有未初始化数据的张量
x= torch.empty(5,3)
print(x)

#随机值 5行三列
y = torch.rand(5,3)
print("yyyyyy:",y)

z = torch.add(x,y)
print(z)
# 直接传值
x = torch.tensor([5.5,3])
print(x)


x = x.new_ones(5,3,dtype=torch.double)
print("xdouble",x)
x = torch.randn_like(x,dtype=torch.float)
print(x)

#张量  举证维度
print(x.size())

# 索引  : ,1   全部第一个数字
print(x[:,1])

# 4*4
x = torch.randn(4,4)
# 改变形状 向量（1行） 16个
y = x.view(16)
# -1代表自动调节  ？ 8   ？=2
z = x.view(-1,8)
print(x.size(),y.size(),z.size())


# 与 Numpy转换
a = torch.ones(5)
b = a.numpy()
print(b)

import numpy as np
# numpy 转换格式 tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
