import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

RAM_ENV_NAME = 'Optimizing_FL_with_RL'

#Agent parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001 # learning rate for Q-network
TAU = 0.001 # step-size for soft update
GAMMA = 0.99 # discount ratio for DDQN



#Training parameters
RAM_NUM_EPISODE = 500 #500
# VISUAL_NUM_EPISODE = 3000
EPS_INIT = 1 # in the epsilon greedy strategy, we need an epsilon, at first, epsilon = 1
EPS_DECAY = 0.94 # at each iteration, epsilon decreases according to the decay ratio = 0.995
EPS_MIN = 0.05 # epsilon can not be infinitely smaller, the minimum epsilon = 0.05
MAX_T = 20 # the maximum steps for each episode is MAX_T
# NUM_FRAME = 2

