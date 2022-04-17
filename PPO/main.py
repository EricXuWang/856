import numpy as np
from utils import *
from agent import *
from config import *
from FL_env import *
import multiprocessing as mp
import matplotlib.pyplot as plt
np.random.seed(3)
def train(agent, env, n_episode, n_update = 4, update_frequency = 1, max_t = 1500, scale = 1):
    rewards_log = []
    average_log = []
    state_history = []
    action_history = []
    done_history = []
    reward_history = []
        
    for i in range(1, n_episode + 1):
        state = env.reset()
        done = False
        t = 0
        if len(state_history) == 0:
            state_history.append(list(state))
        else:
            state_history[-1] = list(state)
        episodic_reward = 0
        
        while not done and t < max_t:
            action = agent.act(state)
            print('\r episode {}, step {}, action {}'.format(i, t, action))
            t += 1
            next_state, reward, done = env.step(action)
            next_state = np.squeeze(next_state)
            episodic_reward += reward
            action_history.append(action)
            done_history.append(done)
            reward_history.append(reward * scale)
            state = next_state
            state_history.append(list(state))
        
        if i % update_frequency == 0:
            states, actions, log_probs, rewards, dones = agent.process_data(state_history, action_history, reward_history, done_history, 10)
            for _ in range(n_update):
                agent.learn(states, actions, log_probs, rewards, dones)
            state_history = []
            action_history = []
            done_history = []
            reward_history = []
        
        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        
        print('\rEpisode {} Reward {:.2f}, Average Reward {:.2f}'.format(i, episodic_reward, average_log[-1]), end='')
        if not done:
            print('\nEpisode {} did not end'.format(i))
        if i % 200 == 0:
            print()
            
    return rewards_log, average_log

def main(q):
    R_log = []
    env = env_fl()
    agent = Agent_discrete(state_size = env.observation_space,
                           action_size = env.action_space,
                           lr = LEARNING_RATE,
                           beta = BETA,
                           eps = EPS,
                           tau = TAU,
                           gamma = GAMMA,
                           device = DEVICE,
                           hidden = HIDDEN_DISCRETE,
                           mode = MODE)
    rewards_log, _ = train(agent = agent,
                           env = env,
                           n_episode = RAM_NUM_EPISODE,
                           n_update = N_UPDATE,
                           update_frequency = UPDATE_FREQUENCY,
                           max_t = MAX_T,
                           scale = SCALE)
    R_log.append(rewards_log)
    q.put(R_log)


if __name__ == '__main__':
    q = mp.Queue()
    for _ in range(7):
        p = mp.Process(target=main,args=(q,))
        p.start()
    Rew_log = []
    for i in range(7):
        Rew_log.append(q.get())
    np.save('{}_results.npy'.format(RAM_DISCRETE_ENV_NAME), Rew_log)
    rewards1000 = np.load('Optimizing_FL_with_PPO_results.npy')
    rewards1000 = rewards1000.reshape(7, 500)
    fig, ax = plt.subplots(dpi=72 * 4)
    ax.fill_between(range(1,501),
            np.quantile(rewards1000, 0.95, axis = 0),
            np.quantile(rewards1000, 0.05, axis = 0),
            color="g",
            alpha=0.2)
    ax.plot(np.mean(rewards1000, axis=0),color = "g")
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.axis(xmin=-10,xmax=500)
    ax.set_xticks(np.arange(0, 600, 100), minor=False)
    ax.set_yticks(np.arange(-12, -1, 1), minor=False)
    # plt.legend(loc=4)
    plt.grid()
    plt.savefig("PPO.png", dpi=300, bbox_inches="tight")
    plt.show()