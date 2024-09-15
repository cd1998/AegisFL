import torch
import torchvision.models as models
import torch.nn.functional as F
from scipy.stats import mode
from torch import optim
import get_har
import get_mnist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import net
import attack
import modified_ckks
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import copy
import numpy as np
seed = 42
torch.manual_seed(seed)

def train_model(dataset,num_clients,num_mal_clients,scaling_factor,local_iteration,global_iteration,attack_type):
    num_clients = num_clients
    num_mal_clients = num_mal_clients
    scaling_factor = scaling_factor
    attack_type = attack_type
    if dataset == "HAR":
        number_label = 6
        model_len = 72710
        X_test, y_test, train_loader, test_loader = get_har.get_data()
        each_worker_data, each_worker_label = get_har.assign_data(train_data=train_loader)
        each_worker_data_du, each_worker_label_du = copy.deepcopy(each_worker_data), copy.deepcopy(each_worker_label)
        aggregation_model = net.Full_connect_har()
        aggregation_optimize = optim.SGD(aggregation_model.parameters(), lr=0.03)

        for model_param in aggregation_model.parameters():
            print('aggregation_model_param',model_param.data)
        print('----------------------------------------------')

        client_model = []
        client_optimize = []
        for i in range(num_clients):
            classfier = net.Full_connect_har()
            classfier.load_state_dict(aggregation_model.state_dict())
            client_model.append(classfier)
            # 创建 优化器
            optimizer = optim.SGD(client_model[i].parameters(), lr=0.03)
            client_optimize.append(optimizer)
    else:
        number_label = 10
        model_len = 101770
        each_worker_data, each_worker_label, X_test, y_test = get_mnist.get_data()
        each_worker_data_du, each_worker_label_du = copy.deepcopy(each_worker_data), copy.deepcopy(each_worker_label)
        aggregation_model = net.Full_connect_mnist()
        aggregation_optimize = optim.SGD(aggregation_model.parameters(), lr=0.03)
        client_model = []
        client_optimize = []
        for i in range(num_clients):
            classfier = net.Full_connect_mnist()
            classfier.load_state_dict(aggregation_model.state_dict())
            client_model.append(classfier)
            # 创建 优化器
            optimizer = optim.SGD(client_model[i].parameters(), lr=0.03)
            client_optimize.append(optimizer)

    # 创建 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    local_iteration = local_iteration
    for epoch in range(local_iteration):
        for i in range(num_clients):
            # 1. 前向传播, 预测输出
            outputs = client_model[i](each_worker_data[i])
            # 2. 计算损失值
            loss = criterion(outputs, each_worker_label[i])
            # 3. 梯度清零
            client_optimize[i].zero_grad()
            # 4. 误差反向传播, 计算梯度并累加
            loss.backward()
            # 5. 更新模型参数
            client_optimize[i].step()

    client_gradient_round_1 = []
    for i in range(num_clients):
        client_gradient_round_1.append(torch.sub(parameters_to_vector(aggregation_model.parameters()),
                                           parameters_to_vector(client_model[i].parameters())))
    global_gradient_vector = sum(client_gradient_round_1) / num_clients
    global_gradient_detach = global_gradient_vector.detach()
    global_gradient_numpy = global_gradient_detach.numpy()

    cipher_aggre_model_1, cipher_aggre_model_2 = modified_ckks.encrypt(
        parameters_to_vector(aggregation_model.parameters()).detach().numpy(), model_len)
    cipher_client_model_round_1 = []
    cipher_client_model_round_2 = []
    for i in range(num_clients):
        cipher1, cipher2 = modified_ckks.encrypt(parameters_to_vector(client_model[i].parameters()).detach().numpy(),
                                                 model_len)
        cipher_client_model_round_1.append(cipher1)
        cipher_client_model_round_2.append(cipher2)

    cipher_client_gradient_round_1 = []
    for i in range(num_clients):
        c = modified_ckks.sub(cipher_aggre_model_1, cipher_client_model_round_1[i])
        cipher_client_gradient_round_1.append(c)

    weight = np.full(num_clients, 1 / num_clients)
    cipher_global_gradient = modified_ckks.weight_sum(cipher_client_gradient_round_1, weight)
    cipher_aggre_model_1 = modified_ckks.modswitch(cipher_aggre_model_1, cipher_global_gradient[1].parms_id())

    cipher_global_model = modified_ckks.sub(cipher_aggre_model_1, cipher_global_gradient)
    plain = modified_ckks.decrypt(cipher_global_model)
    agg_model_vector = np.concatenate(plain)
    print(agg_model_vector)
    agg_model_vector = agg_model_vector[0:model_len]
    a = agg_model_vector.astype(np.float32)
    print(type(a[3]))
    print(len(agg_model_vector))
    agg_gradient_tensor = torch.from_numpy(a)
    agg_gradient_tensor.requires_grad_(True)
    vector_to_parameters(agg_gradient_tensor, aggregation_model.parameters())

    torch.save(aggregation_model.state_dict(), f'model/modified_flame_HE.pt')
    aggregation_model.load_state_dict(torch.load(f'model/modified_flame_HE.pt'))
    for i in range(num_clients):
        client_model[i].load_state_dict(torch.load(f'model/modified_flame_HE.pt'))

    if attack_type == "backdoor":
        each_worker_data_malicious, each_worker_label_malicious = attack.scaling_attack_insert_backdoor(each_worker_data_du, each_worker_label_du, num_clients)
        X_test_with_backdoor, y_test_with_backdoor = attack.add_backdoor(X_test.clone(),y_test.clone())  # 不用clone()会改变X_test,y_test原本的值，因为在python中，对象通常传递的是地址

    if attack_type == "flipping":
        each_worker_label_malicious = attack.label_flipping_attack(each_worker_label_du, num_clients)

    global_iteration = global_iteration
    for round in range(global_iteration):
        random_numbers = np.arange(num_mal_clients)
        for epoch in range(local_iteration):
            for i in range(num_clients):
                if i in random_numbers and attack_type == "backdoor":
                    outputs = client_model[i](each_worker_data_malicious[i])
                    loss = criterion(outputs, each_worker_label_malicious[i])
                    client_optimize[i].zero_grad()
                    loss.backward()
                    client_optimize[i].step()
                elif i in random_numbers and attack_type == "flipping":
                    outputs = client_model[i](each_worker_data[i])
                    loss = criterion(outputs, each_worker_label_malicious[i])
                    client_optimize[i].zero_grad()
                    loss.backward()
                    client_optimize[i].step()
                else:
                    outputs = client_model[i](each_worker_data[i])
                    loss = criterion(outputs, each_worker_label[i])
                    client_optimize[i].zero_grad()
                    loss.backward()
                    client_optimize[i].step()


        client_gradient_round = []
        for i in range(num_clients):
            a = torch.sub(parameters_to_vector(aggregation_model.parameters()),parameters_to_vector(client_model[i].parameters()))
            if i in random_numbers and attack_type == "backdoor":
                a.mul_(scaling_factor)
            if i in random_numbers and attack_type == "ut":
                a.normal_(mean=0,std=1)

            client_gradient_round.append(a)

        stacked = torch.stack(client_gradient_round)
        stacked_tensor = stacked.detach()
        numpy_array = stacked_tensor.numpy()
        # vector_norm = np.linalg.norm(global_gradient_numpy)
        # matrix_norms = np.linalg.norm(numpy_array, axis=1)
        # dot_products = np.dot(numpy_array, global_gradient_numpy)
        # cosine_similarity = dot_products / (matrix_norms * vector_norm)
        # print(cosine_similarity)

        # 计算每一行的范数（模）
        matrix_norms = np.linalg.norm(numpy_array, axis=1)
        # 计算每一行与其他行的点积
        dot_products = np.dot(numpy_array, numpy_array.T)  # 使用矩阵的转置进行点积计算
        # 计算余弦相似度
        cosine_similarity = dot_products / (np.outer(matrix_norms, matrix_norms))

        print(cosine_similarity)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(cosine_similarity)
        #print('label', cluster_labels)

        unique_values, counts = np.unique(cluster_labels, return_counts=True)
        mode_indices = np.where(counts == counts.max())
        modes = unique_values[mode_indices]
        indices = np.where(cluster_labels == modes[0])

        constant_indices = indices[0]
        Len = len(constant_indices)

        euclidean_distances = np.linalg.norm(numpy_array - global_gradient_numpy, axis=1)
        S_t = np.median(euclidean_distances)
        #print('decliean', euclidean_distances)

        client_gradient = []
        for i in constant_indices:
            client_gradient.append(client_gradient_round[i] * np.minimum(1, S_t / euclidean_distances[i]) )

        global_gradient_vector = sum(client_gradient) / Len
        global_gradient_detach = global_gradient_vector.detach()
        global_gradient_numpy = global_gradient_detach.numpy()
        aggregation_model_vector = torch.sub(parameters_to_vector(aggregation_model.parameters()),
                                             global_gradient_vector)
        vector_to_parameters(aggregation_model_vector, aggregation_model.parameters())
        torch.save(aggregation_model.state_dict(), f'model/modified_flame.pt')
        aggregation_model.load_state_dict(torch.load(f'model/modified_flame.pt'))
        for i in range(num_clients):
            client_model[i].load_state_dict(torch.load(f'model/modified_flame.pt'))


        # 预测的输出标签类别 (每行取概率最大的索引为标签类别值)
        outputs = aggregation_model(X_test)
        output_labels = torch.max(outputs, dim=1)[1]

        # 预测准确的数量 (相同位置的类别值相等则为True, 不能则为False, True是1, False是0, 全部结果相加就是正确预测的数量)
        accurate_count = torch.sum(output_labels == y_test)
        # 计算准确率
        accuracy = accurate_count / output_labels.shape[0]
        loss = criterion(outputs, y_test)
        print(f"Round[{round}/{global_iteration}]: loss={loss}, accuracy={accuracy:.3f}")

        if attack_type == "backdoor":
            outputs_backdoor = aggregation_model(X_test_with_backdoor)
            output_labels_backdoor = torch.max(outputs_backdoor, dim=1)[1]
            accurate_count_backdoor = torch.sum(output_labels_backdoor == y_test_with_backdoor)
            accuracy_backdoor = accurate_count_backdoor / output_labels_backdoor.shape[0]
            print(f"backdoor attack success rate = {accuracy_backdoor:.3f}")


if __name__ == "__main__":
    train_model(dataset="HAR",num_clients=30,num_mal_clients=12,scaling_factor=5,local_iteration=10,global_iteration=100,attack_type="flipping")





