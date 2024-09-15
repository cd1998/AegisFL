import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.init as init
from torch.nn.utils import parameters_to_vector
import numpy as np
class Full_connect_har(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 隐含层
        self.hidden = torch.nn.Linear(561,128)
        # 输出层
        self.out = torch.nn.Linear(128,6)

    def forward(self, x) :
        """
        前向传播 (预测输出)
        """
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)

        # x = self.hidden(inputs)  # 数据传输到隐含层
        # outputs = self.out(F.relu(x))  # 应用 ReLU 激活函数, 增加非线性拟合能力, 然后传输到输出层
        # return F.softmax(outputs, dim=1)  # 把输出应用 softmax 激活函数 (把各类别输出值映射为对应的概率)

class Full_connect_mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 隐含层
        self.hidden = torch.nn.Linear(28*28, 128)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        # 输出层
        self.out = torch.nn.Linear(128, 10)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        """
        前向传播 (预测输出)
        """
        # x = x.view(-1, 28*28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)

        return F.softmax(x, dim=1)  # 把输出应用 softmax 激活函数 (把各类别输出值映射为对应的概率)

class test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 隐含层
        self.hidden = torch.nn.Linear(2, 2)
        # 输出层
        self.out = torch.nn.Linear(2, 1)

    def forward(self, x):
        """
        前向传播 (预测输出)
        """
        # x = x.view(-1, 28*28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)

        return F.softmax(x, dim=1)  # 把输出应用 softmax 激活函数 (把各类别输出值映射为对应的概率)

chen = Full_connect_mnist()
for model_param in chen.parameters():
    print('model_param',model_param.data)
print('----------------------------------------------')
vector = parameters_to_vector(chen.parameters()).detach().numpy()
print(type(vector))
print(vector.shape)
print(vector)
print(vector.sum())

# data = np.random.normal(0, 1, 10)  # 生成服从正态分布的随机数据
data = vector
print('data',data)
print('data_sum', data.sum())
# 绘制直方图
plt.hist(data, bins=30, density=True, alpha=0.5, color='blue', edgecolor='black')

# 绘制正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
plt.plot(x, p, 'k', linewidth=2)

plt.show()








