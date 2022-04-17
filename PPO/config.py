import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment names
RAM_DISCRETE_ENV_NAME = 'Optimizing_FL_with_PPO'
CONSTANT = 90

#Agent parameters
LEARNING_RATE = 0.005
GAMMA = 0.99
BETA = 3
EPS = 0.2
TAU = 0.99
MODE = 'TD'
HIDDEN_DISCRETE = [128]


#Training parameters
RAM_NUM_EPISODE = 500
SCALE = 1
MAX_T = 20
NUM_FRAME = 2
N_UPDATE = 10
UPDATE_FREQUENCY = 2
