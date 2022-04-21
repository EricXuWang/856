#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch


# In[2]:


abalone=pd.read_csv('utils/abalone.csv',header=0)
abalone['Sex'] = abalone['Sex'].replace('M',0)
abalone['Sex'] = abalone['Sex'].replace('F',1)
abalone['Sex'] = abalone['Sex'].replace('I',2)
Y = abalone['Sex']           
X = abalone[abalone.columns[1:8]]


# In[3]:


X_train = X[0:round(X.shape[0]*0.4)]
Y_train = Y[0:round(X.shape[0]*0.4)]


# In[4]:


X_test = X[round(X.shape[0]*0.33):round(X.shape[0]*0.66)]
Y_test = Y[round(X.shape[0]*0.33):round(X.shape[0]*0.66)]


# In[5]:


X_reward = X[round(X.shape[0]*0.66) : :]
Y_reward = Y[round(X.shape[0]*0.66) : :]


# In[6]:


Y_train = Y_train.values
Y_test = Y_test.values
Y_reward = Y_reward.values


# In[7]:


X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
X_reward = preprocessing.scale(X_reward)


# In[8]:


X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_reward = X_reward.astype(np.float32)


# In[9]:


X_reward.shape


# In[10]:


Y_reward.shape


# In[ ]:





# In[ ]:





# In[11]:


X_trainTensor = torch.from_numpy(X_train) # convert to tensors
Y_trainTensor = torch.from_numpy(Y_train)
X_testTensor = torch.from_numpy(X_test)
Y_testTensor = torch.from_numpy(Y_test)
X_rewardTensor = torch.from_numpy(X_reward)
Y_rewardTensor = torch.from_numpy(Y_reward)


# In[12]:


X_rewardTensor.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


# '''
# Generate the dictionary of users for test set
# user 0 only holds the Male abalone
# user 1 only holds the Femal abalone
# user 2 only holds the infant abalone
# user 3 holds whole dataset. Therefore, user 3 holds the data with highest quality
# '''
# num_users = 4
# dict_users, all_idxs = {}, [i for i in range(num_users)]
# for i in range(0,num_users-1):
#     dict_users[i] = set(np.where(Y_train == i)[0])
    
# dict_users[3] = set(range((X_train.shape[0])))

num_users = 4
dict_users, all_idxs = {}, [i for i in range(num_users)]
for i in range(0,num_users):
#     dict_users[i] = set(np.where(Y_train == i)[0])
    dict_users[i] = set(range((X_train.shape[0])))


# In[14]:



# '''
# Generate the dictionary of users for train set 
# user 0 only holds the Male abalone
# user 1 only holds the Femal abalone
# user 2 only holds the infant abalone
# user 3 holds whole dataset. Therefore, user 3 holds the data with highest quality
# '''
num_users = 4
dict_users_test, all_idxs_test = {}, [i for i in range(num_users)]
for i in range(0,num_users):
#     dict_users_test[i] = set(np.where(Y_test == i)[0])
    dict_users_test[i] = set(range((X_test.shape[0])))


# In[15]:


# define GetLoader class， inherate Dataset method，write the method __getitem__() and __len__()
class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data, data_label):
        self.data = data
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)
    
'''
train_data are used to train the RL agent
test_data are used to test the RL agent
reward_data are used to calculated the reward in the training and testing stage

'''
train_data = GetLoader(X_trainTensor, Y_trainTensor)
test_data = GetLoader(X_testTensor, Y_testTensor)
reward_data = GetLoader(X_rewardTensor, Y_rewardTensor)


# In[ ]:





# In[ ]:




