
'''
Generate dict_users, train_data, test_data, reward_data

'''
from generate_abalone_data import *

'''
Import dependencies related to federated learning

'''
import copy
import numpy as np
import torch
from models.Update import LocalUpdate
from models.test import test_img
from models.Nets import Logistic_Regression
import math


class env_fl:
    def __init__(self, B=500, E=5):
        print()
        print('============> FL environment Initialization <============')
        self.state = None
        # In mac, map_location=torch.device('cpu') have to be explicitly specified even though it does not have gpu,
        # If you have GPU on Linux and Windows, you can delete it.
        self.Init_env = copy.deepcopy(torch.load('environment/Initial_Env.pth', map_location=torch.device('cpu')))
        self.initial_model_dic = copy.deepcopy(self.Init_env['Initial_model_dic'])
        self.initial_state = copy.deepcopy(self.Init_env['Initial_state'])
        self.w_loc_dic = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.net_glob = None
        self.net_glob = copy.deepcopy(Logistic_Regression().to(self.device))
        return self.state

    def step(self, action):

        # ============================ Calculate the new state according to the current action =========================
        self.net_glob.load_state_dict(self.glob_model)
        local = LocalUpdate(self.B, dataset=self.dataset_train, idxs=self.dict_users[action])

        w_locals_dic, loss = local.train(net=copy.deepcopy(self.net_glob).to(self.device), E=self.E, device=self.device)
        if action != 3:
            w_avg = copy.deepcopy(w_locals_dic)
            for k in w_avg.keys():
                # Calculate the averaged global model
                w_avg[k] = torch.div(w_avg[k], -1)
            w_locals_dic = copy.deepcopy(w_avg)
        self.w_loc_dic[action] = copy.deepcopy(w_locals_dic)
        self.glob_model = copy.deepcopy(w_locals_dic)
        self.w_loc_dic[-1] = copy.deepcopy(w_locals_dic)


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
        acc_test = acc_test * 0.01
        omega = 64
        target_acc = 0.55
        reward = math.pow(omega, acc_test - target_acc) - 1

        print('test accuracy is {}'.format(acc_test))

        # ================================== test if the current episode is done =======================================
        if acc_test < target_acc:
            done = False
        else:
            done = True
        return self.state, reward, done



