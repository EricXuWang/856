from utils.agent import *
from config import *
from FL_env import *
import multiprocessing as mp
import matplotlib.pyplot as plt


np.random.seed(3)
def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    rewards_log = []
    average_log = []
    eps = eps_init
    training_setp_log = 1
    for i in range(1, 1 + num_episode):

        episodic_reward = 0
        done = False
        state = env.reset()
        t = 1

        while not done and t <= max_t:
            print('episode {}, step {}, total training step {}'.format(i, t, training_setp_log))
            t += 1
            training_setp_log +=1
            action = agent.act(state, eps)
            print('\raction {}, eps {}'.format(action,eps))
            next_state, reward, done = env.step(action)
            agent.memory.remember(state, action, reward, next_state, done)
            if t % 2 == 0 and len(agent.memory) >= agent.bs:
                print()
                print('============> Agent update <============')
                agent.learn()
                agent.soft_update(agent.tau)

            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]))#, end='')
        if i % 100 == 0:
            print()
        eps = max(eps * eps_decay, eps_min)


    return rewards_log, average_log


def main(q):
    R_log =[]
    env = env_fl()
    agent = Agent(env.observation_space, env.action_space, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    _,rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    R_log.append(rewards_log)
    q.put(R_log)



if __name__ == '__main__':
    # R_log =[]
    # # init environment
    # env = env_fl()
    # # init agent
    # agent = Agent(env.observation_space, env.action_space, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    # # train
    # _, rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    # # track rewards
    # R_log.append(rewards_log)
    #
    # init queue with multi-process
    q = mp.Queue()
    # q.put(R_log)

    # requre 7 workers
    for _ in range(7):
        p = mp.Process(target=main,args=(q,))
        p.start()


    Rew_log = []
    for i in range(7):
        Rew_log.append(q.get())

    # save result
    np.save('results/{}_results_2.npy'.format(RAM_ENV_NAME), Rew_log)

    # load result
    rewards1000 = np.load(f'results/{RAM_ENV_NAME}_results_2.npy')
    rewards1000 = rewards1000.reshape(7, 500)

    # plot result
    fig, ax = plt.subplots(dpi=72 * 4)

    ax.fill_between(range(1, 501),
                    np.quantile(rewards1000, 0.95, axis=0),
                    np.quantile(rewards1000, 0.05, axis=0),
                    color="g",
                    alpha=0.2)

    ax.plot(np.mean(rewards1000, axis=0), color="g")

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.axis(xmin=-10, xmax=500)
    ax.set_xticks(np.arange(0, 600, 100), minor=False)
    ax.set_yticks(np.arange(-12, -1, 1), minor=False)
    plt.grid()
    plt.savefig("DQN_2.png", dpi=300, bbox_inches="tight")
    plt.show()