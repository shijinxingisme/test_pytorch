import torch
from torch import tensor

# 0:scalar  数值
# 1:vector  向量
# 2:matrix  矩阵
# 3:n-dimensional tensor  高维

# scalar
x = tensor(42.)
print(x)
#维度
print(x.dim())
#计算
print(2*x)
#值
print(x.item())

#vector
v= tensor([1.5,-0.5,3.0])
print(v)
print(v.dim())
print(v.size())

#Matix
M = tensor([[1.,2.],[3.,4.]])
print("a:",M.matmul(M))
print("b:",tensor([1.,0.]).matmul(M))
print("c:",M*M)
print("d:",tensor([1.,2.]).matmul(M))
