import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
RAM_ENV_NAME = 'Optimizing_FL_with_RL'

#Agent parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TAU = 0.001 
GAMMA = 0.99 
DUEL = False

#Training parameters
RAM_NUM_EPISODE = 500
EPS_INIT = 1 
EPS_DECAY = 0.94 
EPS_MIN = 0.05 
MAX_T = 20 


