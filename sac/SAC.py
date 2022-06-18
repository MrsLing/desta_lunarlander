import gym
import torch
import random 
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.distributions import Normal
from env_sac import NormalizedEnv
import os
import pandas as pd
from time import *

class ReplayBeffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(device), \
               torch.FloatTensor(action_list).to(device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state_list).to(device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        print("self.state_dim",self.state_dim)
        self.action_dim = env.action_space.shape[0]
        print("self.action_dim",self.action_dim)
        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.value_net = ValueNet(self.state_dim).to(device)
        self.target_value_net = ValueNet(self.state_dim).to(device)
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = ReplayBeffer(buffer_maxlen)

    def get_action(self, state):
        action = self.policy_net.action(state)
        return action

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        new_action, log_prob = self.policy_net.evaluate(state)

        # V value loss
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
        return np.array([value_loss.detach(),q1_value_loss.detach(),q2_value_loss.detach(),policy_loss.detach()])

def avg(loss_list):
    avg_loss=[]
    step=100
    print(len(loss_list))
    l=len(loss_list)-len(loss_list)%step
    for i in range(0,l,step):
        s=0
        k=i+step
        #print(k)
        for j in range(i,k,1):
            s+=loss_list[j]
        avg_loss.append(s/step)
    return avg_loss
def main(env, agent, Episode,ep_steps,batch_size):
    Return = []
    tra_x=[]
    tra_y=[]
    all_ep_r = []
    v_loss=[]
    q1_loss=[]
    q2_loss=[]
    p_loss=[]
    action_range = [env.action_space.low, env.action_space.high]
    begin_time = time()
    for episode in range(Episode):
        score = 0
        state,sensor_pos= env.reset()
        for i in range(ep_steps):
            if episode==Episode-1:
                tra_x.append(state[0])
                tra_y.append(state[1])
            action = agent.get_action(state)
            # action output range[-1,1],expand to allowable range
            #action_in =  action * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0
            next_state, reward, done, _ ,sensor_pos= env.step(action,i)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state
            score += reward
            #if done:
               # break
            if agent.buffer.buffer_len() > 500:
                loss=agent.update(batch_size)
                v_loss.append(loss[0])
                q1_loss.append(loss[1])
                q2_loss.append(loss[2])
                p_loss.append(loss[3])

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
        Return.append(score)
        if episode == 0: all_ep_r.append(score)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + score*0.1)
    end_time = time()
    run_time = end_time-begin_time
    print ('该循环程序运行时间：',run_time) #该循环程序运行时间： 1.4201874732
    v_loss=avg(v_loss)
    q1_loss=avg(q1_loss) 
    q2_loss=avg(q2_loss)
    p_loss=avg(p_loss)
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/eval/"
    writer = SummaryWriter(log_dir)
    writer=pd.ExcelWriter('rew-tra-sac.xlsx')
    df1=pd.DataFrame(data={'rewards':Return,'moving_average_rewards':all_ep_r})
    df1.to_excel(writer,'rewards')
    #df2=pd.DataFrame(data={'actor_loss':aloss,'critic_loss':closs})
    #df2.to_excel(writer,'loss')
    df2=pd.DataFrame(data={'car_x':tra_x,'car_y':tra_y})
    df2.to_excel(writer,'trajectory')
    df3=pd.DataFrame(data={'v_loss':v_loss,'q1_loss':q1_loss,'q2_loss':q2_loss,'ploss':p_loss})
    df3.to_excel(writer,'loss')
    writer.save()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()
    plt.plot(tra_x,tra_y)
    plt.plot(tra_x,tra_y,">")
    plt.ylabel('trajectory_x')
    plt.xlabel("trajectory_y")
    plt.grid(True)
    plt.show()
    print(tra_x,tra_y)
if __name__ == '__main__':
    #env = gym.make("Pendulum-v1")
    env=NormalizedEnv()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Params
    tau = 0.0005
    gamma = 0.995
    q_lr = 3e-3
    value_lr = 3e-4
    policy_lr = 1e-4
    buffer_maxlen = 100000

    Episode = 500
    ep_steps=200
    batch_size = 128

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
    main(env, agent, Episode,ep_steps, batch_size)