#!/usr/bin/env python
# coding: utf-8


'''
Generate dict_users, train_data, test_data, reward_data

'''
from generate_abalone_data import *

'''
Import dependencies related to federated learning

'''
import copy
import numpy as np
# from torchvision import datasets
import torch
# from utils.data_select import data_selection
# from utils.model_select import model_selection
# from utils.client_select import client_selection
from models.Update import LocalUpdate
# from models.Fed import FedAvg
from models.test import test_img
# import random
from models.Nets import Logistic_Regression
import math


# In[2]:

#
# # B is the local mini batch size
# B = 50
# # E is the local epochs
# E = 5
# # model, cnn or mlp
# Model = 'CifarNet'
# # Lr is the learning rate of SGD
# Lr = 0.1
# # dataset is mnist or CIFAR
# dataset = 'CIFAR'
# # data_distribution = 'IID'
# # data_distribution = 'Non_IID_two_classes'
# data_distribution = 'Non_IID'
# non_iid_level = 80


# In[3]:


class env_fl:

    def __init__(self, B=500, E=5):
        # self.dataset = dataset
        # self.data_distribution = data_distribution
        # self.non_iid_level = non_iid_level
        # self.Model = Model
        # self.B = B
        # self.E = E
        # self.Lr = Lr
        # self.num_users = 100
        # self.state = None
        # self.num_dim_after_reduction = 60
        # self.pca = PCA(n_components = self.num_dim_after_reduction )
        # self.glob_model = None
        # self.action = None
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.dataset_train = None
        # self.dataset_test = None
        # self.dict_users = None
        # self.net_glob = None
        # self.w_loc_dic = []
        # self.observation_space = (self.num_users+1) * self.num_dim_after_reduction
        # self.action_space = self.num_users
        # self.initial_w_local_dic = []
        # self.initial_state = []
        # self.initial_globel_model = None
        print()
        print('============> FL environment Initialization <============')
        self.state = None
        self.Init_env = copy.deepcopy(torch.load('Initial_Env.pth'))
        self.initial_model_dic = copy.deepcopy(self.Init_env['Initial_model_dic'])
        self.initial_state = copy.deepcopy(self.Init_env['Initial_state'])
        self.w_loc_dic = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.net_glob = Logistic_Regression().to(self.device)
        self.glob_model = None
        self.dataset_train = train_data
        self.dataset_test = reward_data
        self.dict_users = dict_users
        self.B = B
        self.E = E
        self.observation_space = 24 * 5
        self.action_space = 4
        self.net_glob = None

    def reset(self):

        print()
        print('============> FL environment reset <============')
        self.w_loc_dic = copy.deepcopy(self.initial_model_dic)
        self.state = copy.deepcopy(self.initial_state)
        self.glob_model = copy.deepcopy(self.w_loc_dic[-1])
        # print(self.initial_model_dic)
        self.net_glob = None
        self.net_glob = copy.deepcopy(Logistic_Regression().to(self.device))
        return self.state

    def step(self, action):

        # ============================ Calculate the new state according to the current action =========================
        # t1 = time.perf_counter()
        self.net_glob.load_state_dict(self.glob_model)
        # for act in action:

        local = LocalUpdate(self.B, dataset=self.dataset_train, idxs=self.dict_users[action])

        w_locals_dic, loss = local.train(net=copy.deepcopy(self.net_glob).to(self.device), E=self.E, device=self.device)


        if action != 3:
            w_avg = copy.deepcopy(w_locals_dic)
            for k in w_avg.keys():
                # Calculate the averaged global model
                w_avg[k] = torch.div(w_avg[k], -1)
            w_locals_dic = copy.deepcopy(w_avg)


        # t2= time.perf_counter()
        # w_loc_log.append(w_locals_dic)

        self.w_loc_dic[action] = copy.deepcopy(w_locals_dic)
        # t3 = time.perf_counter()
        # w_glob_log = w_loc_log

        self.glob_model = copy.deepcopy(w_locals_dic)
        # t4 = time.perf_counter()
        self.w_loc_dic[-1] = copy.deepcopy(w_locals_dic)
        # t5 = time.perf_counter()

        # print(self.w_loc_dic[-1])
        # print(len(self.w_loc_dic))
        # print(self.w_loc_dic[-1].keys())

        state = []
        for j in self.w_loc_dic:
            a = []
            for i in j.keys():
                x = j[i].view(-1).cpu().numpy()
                a = np.concatenate((a, x), axis=0)
            state.append(a)
        self.state = np.array(state).reshape(1, -1)

        # ================================= Calculate reward ===========================================================

        self.net_glob.load_state_dict(self.glob_model)
        acc_test = test_img(self.net_glob, self.dataset_test)
        # print(acc_test)

        # t8 = time.perf_counter()
        acc_test = acc_test * 0.01
        omega = 64
        # reward = []
        target_acc = 0.55
        reward = math.pow(omega, acc_test - target_acc) - 1

        print('test accuracy is {}'.format(acc_test))

        # ================================== test if the current episode is done =======================================
        if acc_test < target_acc:
            done = False
        else:
            done = True
        # t9 = time.perf_counter()
        # print('model training {}'.format(t2-t1))
        # print('several deepcopy {}'.format(t5 - t2))
        # print('model flatten {}'.format(t6 - t5))
        # print('PCA transfrom {}'.format(t7 - t6))
        # print('model test{}'.format(t8-t7))
        # print('total time{}'.format(t9 - t1))

        return self.state, reward, done



