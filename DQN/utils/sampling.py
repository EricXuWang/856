#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import random
import copy


# def same_seeds(seed):
#     # Python built-in random module
#     random.seed(seed)
#     # Numpy
#     np.random.seed(seed)
#     # Torch
#     torch.manual_seed(seed)
#     # Cuda
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
#
# same_seeds(19530615)


def generate_iid_data(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # The number of pictures that each device has
    num_items = int(len(dataset)/num_users)
    
    # dict_users = {}, all_idxs = [i for i in range(len(dataset))]
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        # Choose randomly num_items index from all_idexs
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # Delet those indexes that has been selected from candidate indexes
        all_idxs = list(set(all_idxs) - dict_users[i])
    # Return a dict of image index for each device
    return dict_users


def generate_non_iid_Half(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # There are 60000 pictures in the mnist dataset
    num_shards = 200
    num_imgs = int(len(dataset)/200)

    # The index of each shard
    idx_shard = [i for i in range(num_shards)]

    # Set an dict. key is the index of each device, value is empty array
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Set index for each picture in the mnist dataset
    idxs = np.arange(num_shards * num_imgs)
    # Convert labels in tensor into labels in array
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    # Stack the index and its corresponding labels by row
    idxs_labels = np.vstack((idxs, labels))
    # Sort labels as follows: the first row is the index of pictures, the second row is the label fro 0 to 9
    # array([[30207,  5662, 55366, ..., 23285, 15728, 11924],
    # [    0,     0,     0, ...,     9,     9,     9]])
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # Set index as follows:
    # array([[30207,  5662, 55366, ..., 23285, 15728, 11924],
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        # Select randomly 2 index from idx_shard
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # Delet those indexes that has been selected from candidate indexes
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            # concatenate two shard and assign them into one devices
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    # Return a dict of image index for each device
    return dict_users




def generate_non_iid_data(dataset, num_users,non_iid_level):

    gama = non_iid_level
    num_class = 10

    # if dataset == 'mnist':
    #     # Calculate the scaled mean and std of the dataset in order to normalize the dataset.
    #     # It is noted that the python will not conduct normalization until the Dataloader works
    #     # In terms of the path, ../ represents previous level folder, ./ represents the current folder
    #     trainset = datasets.MNIST('../data/mnist/', train=True, download=True)
    #     # print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(trainset.data.min(), trainset.data.max()))
    #     # print(
    #     #     'Mean Pixel Value {} \nPixel Values Std: {}'.format(trainset.data.float().mean(), trainset.data.float().std()))
    #     # print('Scaled Mean Pixel Value {} \nScaled Pixel Values Std: {}'.format(trainset.data.float().mean() / 255,
    #     #                                                                         trainset.data.float().std() / 255))
    #     mean = trainset.data.float().mean() / 255
    #     std = trainset.data.float().std() / 255
    #     # Pictures in the mnist are gray images with one channel
    #     # Normalize with mean 0.1307 and std 0.3081, (0.1307,) means that it is a tuple with one element
    #     # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
    #                                    transform=transforms.Compose([transforms.ToTensor(),
    #                                                                  transforms.Normalize((mean,), (std,))]))
    #     dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
    #                                   transform=transforms.Compose([transforms.ToTensor(),
    #                                                                 transforms.Normalize((mean), (std,))]))
    #
    # elif dataset == 'cifar':
    #     # Calculate mean and std
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True,
    #                                      transform=transforms.Compose([transforms.ToTensor()]))
    #     ldr_train = DataLoader(dataset_train, batch_size=int(len(dataset_train) * 1), shuffle=True)
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     train = iter(ldr_train).next()[0]  # 一个batch的数据
    #     mean = np.mean(train.numpy(), axis=(0, 2, 3))
    #     std = np.std(train.numpy(), axis=(0, 2, 3))
    #
    #     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #
    # elif dataset == 'Fashionmnist':
    #     trainset = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True)
    #     mean = trainset.data.float().mean() / 255
    #     std = trainset.data.float().std() / 255
    #
    #     dataset_train = datasets.FashionMNIST('../data/fashion_mnist/', train=True, download=True,
    #                                           transform=transforms.Compose([transforms.ToTensor(),
    #                                                                         transforms.Normalize((mean,), (std,))]))
    #     dataset_test = datasets.FashionMNIST('../data/fashion_mnist/', train=False, download=True,
    #                                          transform=transforms.Compose([transforms.ToTensor(),
    #                                                                        transforms.Normalize((mean), (std,))]))

    # dataset = dataset_train

    # num_users = 100
    # Set an dict. key is the index of each device, value is empty array
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # Set index for each picture in the mnist dataset
    idxs = np.arange(len(dataset))
    # Convert labels in tensor into labels in array
    # labels = dataset.targets.numpy()

    labels = np.array(dataset.targets)
    # Stack the index and its corresponding labels by row
    idxs_labels = np.vstack((idxs, labels))
    # sort data and its labels in ascending order [0,..,0,1,..,1..,9..,9]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    # gama is the levels of Non-IID data
    # gama = 60
    # num_class = 10
    num_noniid_data = int(len(dataset) / num_users * gama / 100)
    num_iid = int(len(dataset) / num_users * (100 - gama) / 100)
    dict_data = {i: np.array([], dtype='int64') for i in range(num_class)}
    for i in range(num_class):
        dict_data[i] = idxs_labels[0, np.where(idxs_labels[1, :] == i)[0]]

    num_noniid_data = int(len(dataset) / num_users * gama / 100)
    for i in range(len(dict_users)):
        rnd_label = i % (num_class)
        dict_users[i] = dict_data[rnd_label][0:num_noniid_data]
        x = dict_data[rnd_label].tolist()
        del x[0:num_noniid_data]
        dict_data[rnd_label] = np.array(x)

    # num_rand = int(len(dataset) / num_users * (100 - gama) / 100)

    my_copy = copy.deepcopy(dict_data)
    for k in dict_users:
        # my_copy = copy.deepcopy(dict_data)
        candi = my_copy.pop(k % (num_class))
        random_samples = []
        for l in my_copy:
            random_samples = np.concatenate((random_samples, my_copy[l]), axis=0)

        random_samples = random_samples.astype(np.int64)
        random_samples = np.array(random_samples)
        np.random.shuffle(random_samples)

        dict_users[k] = np.concatenate((dict_users[k], random_samples[0:num_iid]), axis=0)
        for i in my_copy:
            for j in random_samples[0:num_iid]:
                if len(np.where(my_copy[i] == j)) != 0:
                    my_copy[i] = np.delete(my_copy[i], [np.where(my_copy[i] == j)])

        my_copy = {**{k % (num_class): candi}, **my_copy}

    dict_users[num_users - 2] = np.concatenate((dict_users[num_users - 2],
                                                my_copy[k % (num_class)]), axis=0)

    while len(dict_users[num_users - 1]) != len(dict_users[0]):
        ind_exchange = random.randint(0, len(dict_users[num_users - 2])-1)

        if idxs_labels[1][np.where(idxs_labels[0] == dict_users[num_users - 2][ind_exchange])] != 9 and idxs_labels[1][
            np.where(idxs_labels[0] == dict_users[num_users - 2][ind_exchange])] != 8:
            #print(idxs_labels[1][np.where(idxs_labels[0] == dict_users[num_users - 2][ind_exchange])])
            dict_users[num_users - 1] = np.concatenate((dict_users[num_users - 1],
                                                        [dict_users[num_users - 2][ind_exchange]]), axis=0)
            dict_users[num_users - 2] = np.delete(dict_users[num_users - 2], [ind_exchange])

    for i in dict_users:
        dict_users[i] = set(dict_users[i])

    return dict_users
# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     # The number of pictures that each device has
#     num_items = int(len(dataset)/num_users)
#     # dict_users = {}, all_idxs = [i for i in range(len(dataset))]
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         # Choose randomly num_items index from all_idexs
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         # Delet those indexes that has been selected from candidate indexes
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     # Return a dict of image index for each device
#     return dict_users













# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)
