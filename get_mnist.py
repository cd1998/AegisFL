import torch
from torchvision import datasets, transforms
import numpy as np

# 设置随机种子以保证结果可重现
torch.manual_seed(0)

# 下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

def get_data():
    # 转换成torch.tensor
    x_train = train_dataset.data.type(torch.FloatTensor)
    x_train = x_train.view(60000,-1)
    y_train = train_dataset.targets
    x_test = test_dataset.data.type(torch.FloatTensor)
    x_test = x_test.view(10000,-1)
    y_test = test_dataset.targets

    # 分配数据给30个客户端
    num_clients = 30
    num_items_per_client = len(train_dataset) // num_clients

    each_worker_data = []
    each_worker_label = []

    for i in range(num_clients):
        start_idx = i * num_items_per_client
        end_idx = start_idx + num_items_per_client
        each_worker_data.append(x_train[start_idx:end_idx])
        each_worker_label.append(y_train[start_idx:end_idx])
    return each_worker_data, each_worker_label, x_test, y_test

if __name__ == "__main__":
    each_worker_data, each_worker_label, x_test, y_test = get_data()