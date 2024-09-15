import numpy
import torch
import torchvision
import torch.utils.data
import pandas as pd
import numpy as np
seed = 42
np.random.seed(seed)
# torch.manual_seed(seed)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data():
    df_train = pd.read_csv('data/HAR/test.csv')
    X_train = df_train.iloc[:, :-2]  # all columns except the target variable
    y_train = df_train.iloc[:, -1]  # records of the target variable only , Activity
    # print(y_train)
    # print(X_train)
    y_train = y_train.apply(lambda x: 0 if x == "STANDING" else (1 if x == 'SITTING' else (2 if x == 'LAYING' else (3 if x == 'WALKING' else (4 if x == 'WALKING_DOWNSTAIRS' else 5)))))
    #print(y_train)

    df_test = pd.read_csv('data/HAR/train.csv')
    X_test = df_test.iloc[:, :-2]  # all columns except the target variable
    y_test = df_test.iloc[:, -1]  # records of the target variable only , Activity
    y_test = y_test.apply(lambda x: 0 if x == "STANDING" else (1 if x == 'SITTING' else (
        2 if x == 'LAYING' else (3 if x == 'WALKING' else (4 if x == 'WALKING_DOWNSTAIRS' else 5)))))

    # SC = StandardScaler()
    # X_train_Scaled = SC.fit_transform(X_train)
    # X_test_Scaled = SC.fit_transform(X_test)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    clients = np.concatenate((df_train.iloc[:, -2].ravel(), df_test.iloc[:, -2].ravel()))

    y_train, y_test, X_train, X_test = [], [], [], []
    for client in range(1, 31):
        mask = tuple([clients == client])
        x_client = X[mask]
        y_client = y[mask]
        #print(x_client.shape)

        split = np.concatenate((np.ones(int(np.ceil(0.75 * len(y_client))), dtype=bool),
                                np.zeros(int(np.floor(0.25 * len(y_client))), dtype=bool)))

        np.random.shuffle(split)  # Generate mask for train test split with ~0.75 1
        x_train_client = x_client[split]
        y_train_client = y_client[split]
        x_test_client = x_client[np.invert(split)]
        y_test_client = y_client[np.invert(split)]
        #print('X_shape',x_train_client.shape)

        # Attach vector of client id to training data for data assignment in assign_data()
        x_train_client = np.insert(x_train_client, 0, client, axis=1) #0表示插入到第一个位置，axis表示沿着水平轴插入

        if len(X_train) == 0:
            X_train = x_train_client
            X_test = x_test_client
            y_test = y_test_client
            y_train = y_train_client
        else:
            X_train = np.append(X_train, x_train_client, axis=0)
            X_test = np.append(X_test, x_test_client, axis=0)
            y_test = np.append(y_test, y_test_client)
            y_train = np.append(y_train, y_train_client)

    tensor_train_X = torch.tensor(X_train, dtype=torch.float32)
    tensor_train_y = torch.tensor(y_train, dtype=torch.int64)
    tensor_test_X = torch.tensor(X_test, dtype=torch.float32)
    tensor_test_y = torch.tensor(y_test, dtype=torch.int64)
    print(tensor_train_X.shape)
    # print('ccc',tensor_test_y)
    print(tensor_test_X.shape)
    # print(tensor_test_y.shape)

    train_dataset = torch.utils.data.TensorDataset(tensor_train_X, tensor_train_y)
    test_dataset = torch.utils.data.TensorDataset(tensor_test_X, tensor_test_y)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    # for chen,dong in train_loader:
    #     print(chen)
    #     print(dong)

    return tensor_test_X, tensor_test_y, train_loader, test_loader


def assign_data(train_data):
    # assign training data to each worker

    each_worker_data = [[] for _ in range(30)]    #[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    each_worker_label = [[] for _ in range(30)]

    for _, (data, label) in enumerate(train_data):

        for (x, y) in zip(data, label):
            clientId = int(x[0].item())-1
            x = x[1:len(x)]
            x = x.reshape(1, 561)

            each_worker_data[clientId].append(x)
            each_worker_label[clientId].append(y)

    each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.stack(each_worker, dim=0) for each_worker in each_worker_label]


    return each_worker_data, each_worker_label
if __name__ == "__main__":
    tensor_test_X, tensor_test_y, train_loader,test_loader = get_data()
    assign_data(train_data=train_loader)
