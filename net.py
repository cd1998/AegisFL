import time
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
        # 输出层
        self.out = torch.nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播 (预测输出)
        """
        # x = x.view(-1, 28*28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)  # 把输出应用 softmax 激活函数 (把各类别输出值映射为对应的概率)













