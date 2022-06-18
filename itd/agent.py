#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2020-09-02 01:19:13
@Discription:
@Environment: python 3.7.7
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import Actor, Critic
from memory import ReplayBuffer

class DDPG:
    def __init__(self, n_states, n_actions, hidden_dim=30, device="cpu", critic_lr=1e-4,
                 actor_lr=1e-5, gamma=0.8, soft_tau=1e-2, memory_capacity=100000, batch_size=256):
        self.device = device
        self.critic = Critic(n_states, n_actions, hidden_dim).to(device)
        self.actor = Actor(n_states, n_actions, hidden_dim).to(device)
        self.target_critic = Critic(n_states, n_actions, hidden_dim).to(device)
        self.target_actor = Actor(n_states, n_actions, hidden_dim).to(device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.memory = ReplayBuffer(memory_capacity)
        self.batch_size = batch_size
        self.soft_tau = soft_tau
        self.gamma = gamma
        # self.sigma0=1.
        # self.sigma1=10.
        # self.sigma2=10.
    # def action_zoom(self,action,min,max):
    #     m=0.5*(action+1)*(max-min)+min
    #     return m
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # torch.detach()用于切断反向传播
        # print("agent里的 action:------",action)
        action=action.detach().cpu().numpy()[0]
        # action[0]=action[0]+1
        # action[0]=np.clip(action[0],0,1)
        # action[1]=np.clip(action[1],-1.,1.)
        # action[0]=self.action_zoom(action[0],0.,2.)
        # action[1]=self.action_zoom(action[1],-180.,180.)
        # action[2]=self.action_zoom(action[2],200.,1000.)
        # print("改变后的action:------",action[0])
        # print("改变后的action:------",action[1])
        return action
        # return action.detach().cpu().numpy()[0]

    def update(self):
    

        if len(self.memory) < self.batch_size:
            return np.array([0,0])
        state, action, reward, next_state, done = self.memory.sample(
            self.batch_size)
        # 将所有变量转为张量
        # self.sigma0*=0.999995
        # self.sigma1*=0.999991
        # self.sigma2*=0.999991
        value_loss=[]
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # 注意critic将(s_t,a)作为输入
        policy_loss = self.critic(state, self.actor(state))

        policy_loss = -policy_loss.mean()
        

        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        loss1=policy_loss.detach()
        loss2=value_loss.detach()
        #print("np.array([loss1,loss2])",np.array([loss1,loss2]))
        return np.array([loss1,loss2])
        # print("sigma0",self.sigma0)
        # print("sigma1",self.sigma1)
        # print("sigma2",self.sigma2)
    def save_model(self, path):
        torch.save(self.target_actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))