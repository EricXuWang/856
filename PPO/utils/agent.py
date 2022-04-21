import torch
import torch.optim as optim
import numpy as np
from utils.networks import *
import math

class Agent_discrete:
    
    def __init__(self, state_size, action_size, lr, beta, eps, tau, gamma, device, hidden=[128], mode='MC'):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.mode = mode

        self.Actor = Actor_discrete(state_size, action_size, hidden).to(self.device)
        self.Critic = Critic(state_size, hidden).to(self.device)
        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr)
            
    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32).view(-1, self.state_size).to(self.device)
            log_probs = self.Actor(states)
            probs = log_probs.exp().view(-1).cpu().numpy()
            action = np.random.choice(a=self.action_size, size=1, replace=False, p=probs)[0]
        return action
    
    def process_data(self, states, actions, rewards, dones, batch_size):
        
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device).view(-1, 1)
        
        #calculate log probabilities and state values
        N = states.size(0)

        log_probs = torch.zeros((N, self.action_size)).to(self.device)
        step = math.floor(N / batch_size)

        
        for ind in range(step):

            output = self.Actor(states[ind*batch_size:(ind+1)*batch_size, :])
            output = np.squeeze(output)
            log_probs[ind*batch_size:(ind+1)*batch_size, :] = output 
        
        log_probs = log_probs[:-1, :]
        
        #calculate discounted rewards, gamma^t r_t
        rewards = np.array(rewards)
        
        return states, actions, log_probs.detach(), rewards, dones
    
    def learn(self, states, actions, log_probs, rewards, dones):

        new_log_probs = self.Actor(states)
        state_values = self.Critic(states)
        new_log_probs = new_log_probs[:-1, :]

        KL_Loss = log_probs.exp() * (log_probs - new_log_probs)
        KL_Loss = KL_Loss.sum(dim=1, keepdim=True)

        log_probs = torch.gather(log_probs, dim=1, index=actions)
        new_log_probs = torch.gather(new_log_probs, dim=1, index=actions)
        
        L = rewards.shape[0]
        with torch.no_grad():
            G = []
            return_value = 0
            if self.mode == 'MC':
                for i in range(L-1, -1, -1):
                    return_value = rewards[i] + self.gamma * return_value * (1-dones[i])
                    G.append(return_value)
                G = G[::-1]
                G = torch.tensor(G, dtype=torch.float).view(-1, 1).to(self.device)
            else:
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
                G = rewards + (1 - dones) * self.gamma * state_values[1:, :]
            
        Critic_Loss = 0.5*(state_values[:-1, :] - G).pow(2).mean()
        
        with torch.no_grad():

            G = G - state_values[:-1, :] # advantage
            for i in range(L-2, -1, -1):
                G[i] += G[i+1]*self.gamma*(1-dones[i])*self.tau # cumulated advantage
        
        ratio = (new_log_probs - log_probs).exp()
        Actor_Loss1 = ratio * G
        Actor_Loss2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * G
        Actor_Loss = -torch.min(Actor_Loss1, Actor_Loss2)
        Actor_Loss += self.beta * KL_Loss
        
        Actor_Loss = Actor_Loss.mean()
        

        self.critic_optimizer.zero_grad()
        Critic_Loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), 1)
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        Actor_Loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 1)
        self.actor_optimizer.step()
            
