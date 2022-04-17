import config
from utils import SumTree

import numpy as np
import random
import bisect
import torch



class Replay_Buffer:
    '''
    Vanilla replay buffer
    '''
    
    def __init__(self, capacity, batch_size=None):
        
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)] # save tuples (state, action, reward, next_state, done)
        self.ind_max = 0 # how many transitions have been stored
        
    def remember(self, state, action, reward, next_state, done):
        
        ind = self.ind_max % self.capacity
        self.memory[ind] = (state, action, reward, next_state, done)
        self.ind_max += 1
        
    def sample(self, k):
        '''
        return sampled transitions. Make sure that there are at least k transitions stored before calling this method 
        '''
        index_set = random.sample(list(range(len(self))), k)
        states = torch.from_numpy(np.vstack([self.memory[ind][0] for ind in index_set])).float()
        actions = torch.from_numpy(np.vstack([self.memory[ind][1] for ind in index_set])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[ind][2] for ind in index_set])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[ind][3] for ind in index_set])).float()
        dones = torch.from_numpy(np.vstack([self.memory[ind][4] for ind in index_set]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return min(self.ind_max, self.capacity)
        
