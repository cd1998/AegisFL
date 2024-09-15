import numpy as np
import torch
import random
import math
def label_flipping_attack(client_label, f):
    num_labels = 6
    for i in range(f):
        client_label[i] = num_labels - client_label[i] - 1
    return client_label

def scaling_attack_insert_backdoor(each_worker_data, each_worker_label, f):
    attacker_chosen_target_label = 2
    for i in range(f):
        p = 0.2
        number_of_backdoored_images = math.ceil(p * each_worker_data[i].size(dim=0))
        benign_images = each_worker_data[i].size(dim=0)
        # expand list of images with number of backdoored images and copy all benign images
        expanded_data = torch.zeros(benign_images + number_of_backdoored_images,
                                    each_worker_data[i].size(dim=1))
        for n in range(benign_images):
            expanded_data[n] = each_worker_data[i][n]

        # duplicate images and add pattern trigger
        for j in range(number_of_backdoored_images):
            # Currently first image is selected every time
            random_number = random.randrange(0, each_worker_data[i].size(dim=0))
            backdoor = each_worker_data[i][random_number, :]
            for k in range(len(backdoor)):
                if (k + 1) % 20 == 0:
                    backdoor[k] = 0
            expanded_data[benign_images + j] = backdoor

        # replace data of compromised worker with expanded data
        each_worker_data[i] = expanded_data

        # expand list of labels with number of backdoored images with attacker chosen target label
        each_worker_label[i] = torch.tensor(each_worker_label[i].tolist() +
                               [attacker_chosen_target_label for i in range(number_of_backdoored_images)])

    return each_worker_data, each_worker_label

def add_backdoor(data, labels):
    attacker_chosen_target_label = 2

    # add pattern trigger
    for i in range(data.size(dim=0)):
        for k in range(data.size(dim=1)):
            if (k + 1) % 20 == 0:
                data[i][k] = 0

        # expand list of labels with number of backdoored images with attacker chosen target label
        for i in range(len(labels)):
            labels[i] = attacker_chosen_target_label


    return data, labels