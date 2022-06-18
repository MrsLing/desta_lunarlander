#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-11 20:58:21
@LastEditor: John
LastEditTime: 2020-10-15 21:23:39
@Discription:
@Environment: python 3.7.7
'''
from token import NUMBER
from typing import Sequence
import torch
import gym 
from agent import DDPG
# from env import NormalizedActions
from noise import OUNoise
import os
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
from env import PendulumEnv
import matplotlib.pyplot as plt
import random    
import math#from xlwt import *
from openpyxl import load_workbook
#需要xlwt库的支持
#import xlwt 
#import xlwings as xw
from openpyxl import Workbook
import pandas as pd
from time import *

import xlwt
def saveEnergy(rows):
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("energy")
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            sheet.write(j, i, rows[i][j])
    workbook.save("energy-non-k.xlsx")
SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/result/"+SEQUENCE+'/'

def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int)  # 1 表示训练，0表示只进行eval
    parser.add_argument("--gamma", default=0.9,
                        type=float)  # q-learning中的gamma
    parser.add_argument("--critic_lr", default=1e-4, type=float)  # critic学习率
    parser.add_argument("--actor_lr", default=1e-5, type=float)
    parser.add_argument("--memory_capacity", default=1000000,
                        type=int, help="capacity of Replay Memory")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="batch size of memory sampling")
    parser.add_argument("--train_eps", default=200, type=int)
    parser.add_argument("--train_steps", default=200, type=int)
    parser.add_argument("--eval_eps", default=200, type=int)  # 训练的最大episode数目
    parser.add_argument("--eval_steps", default=200,
                        type=int)  # 训练每个episode的长度
    parser.add_argument("--target_update", default=4, type=int,
                        help="when(every default 10 eisodes) to update target net ")
    config = parser.parse_args()
    return config


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
def train(cfg):
    print('Start to train ! \n')
    # env = NormalizedActions(gym.make("Pendulum-v0"))
    env=PendulumEnv(on_mouse=True,random_goal=False)
    env.reset()
    # 增加action噪声
    ou_noise = OUNoise(env.action_space)
    # print("env.action_space.................",env.action_space)
    # print("ou_noise::::::::",ou_noise)

    n_states = env.observation_space.shape[0]	
    n_actions = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPG(n_states, n_actions, device="cpu", critic_lr=1e-4,
                 actor_lr=1e-5, gamma=0.9, soft_tau=1e-2, memory_capacity=1000000, batch_size=256)
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    episode=[]
    sensor_x=[]
    sensor_y=[]
    actor_loss=[]
    critic_loss=[]
    total_energy=[]
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    # cfg.train_eps+1s   
    begin_time = time()
    for i_episode in range(1,501):
        state,sensor_pos = env.reset()
        
        ou_noise.reset()
        ep_reward = 0
        ep_energy=[]
        for i_step in range(1, cfg.train_eps+1):
            # print("起始状态---------------",state)
            """
            if i_episode<=100:
                ep_energy.append(state[2])
            if i_episode ==1:
                energy1.append(state[2])
            elif i_episode ==30:
                energy2.append(state[2])
            elif i_episode ==60:
                energy3.append(state[2])
            elif i_episode ==100:
                energy4.append(state[2])"""
            action = agent.select_action(state)
            action= ou_noise.get_action(action)  # 即paper中的random process
            next_state, reward, done, _,sensor_pos = env.step(action,i_step) # 更新环境参数
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)# 将state等这些transition存入memory
            state = next_state# 跳转到下一个状态
            loss=agent.update()
            if loss.all():
                actor_loss.append(loss[0])
                critic_loss.append(loss[1])
            if done:
                break                                  
        total_energy.append(ep_energy)
        print('Episode:', i_episode, ' Reward: %i' %int(ep_reward), 'n_steps:', i_step )
        
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        episode.append(i_episode)
        
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode',
                          ep_steps[-1], i_episode)
    end_time = time()
    run_time = end_time-begin_time
    print ('该循环程序运行时间：',run_time) #该循环程序运行时间： 1.4201874732

    aloss=avg(actor_loss)
    closs=avg(critic_loss)  

    writer=pd.ExcelWriter('rewards_Multi_energy.xlsx')
    df1=pd.DataFrame(data={'rewards':rewards,'moving_average_rewards':moving_average_rewards})
    df1.to_excel(writer,'rewards')
    df2=pd.DataFrame(data={'actor_loss':aloss,'critic_loss':closs})
    df2.to_excel(writer,'loss')
    #df3=pd.DataFrame(data={"energy":e for e in total_energy})
    #df3.to_excel(writer,'energy')
    writer.save()
    plt.plot(rewards)
    plt.show()
    # saveEnergy(total_energy) 
    writer.close()
    print('Complete training！')
    ''' 保存模型 '''
    if not os.path.exists(SAVED_MODEL_PATH): # 检测是否存在文件夹
        os.makedirs(SAVED_MODEL_PATH)
    agent.save_model(SAVED_MODEL_PATH+'checkpoint.pth')
    '''存储reward等相关结果'''
    if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
        os.mkdir(RESULT_PATH)
    np.save(RESULT_PATH+'rewards_train.npy', rewards)
    np.save(RESULT_PATH+'moving_average_rewards_train.npy', moving_average_rewards)
    np.save(RESULT_PATH+'steps_train.npy', ep_steps)

def eval(cfg, saved_model_path = SAVED_MODEL_PATH):
    print('start to eval ! \n')
    
    env=PendulumEnv()
    # env = NormalizedActions(gym.make("Pendulum-v0"))
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states, n_actions, critic_lr=1e-4,
                 actor_lr=1e-5, gamma=0.9, soft_tau=1e-2, memory_capacity=1000000, batch_size=256)
    agent.load_model(saved_model_path+'checkpoint.pth')
    rewards = []
    moving_average_rewards = []
    ep_steps = []
    x_val=[]
    y_val=[]
    sensor_0=[]
    sensor_1=[]
    energy=[]
    log_dir=os.path.split(os.path.abspath(__file__))[0]+"/logs/eval/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_episode in range(1,501):
        state,sensor_pos = env.reset()  # reset环境状态
        ep_reward = 0
        for i_step in range(1, cfg.train_eps+1):
            if i_episode ==500:
                x_val.append(state[0])
                y_val.append(state[1])
                energy.append(state[2])
                sensor_0.append(sensor_pos[0])
                sensor_1.append(sensor_pos[1])
            action = agent.select_action(state)  # 根据当前环境state选择action
            next_state, reward, done, _,sensor_pos = env.step(action,i_step)  # 更新环境参数
            ep_reward += reward
            state = next_state  # 跳转到下一个状态
            if done:
                break
        # print('Episode:', i_episode, ' Reward: %i' %int(ep_reward), 'n_steps:', i_step, 'done: ', done)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
            # print("--------滑动平均",moving_average_rewards[-1])
        writer.add_scalars('rewards',{'raw':rewards[-1], 'moving_average': moving_average_rewards[-1]}, i_episode)
        writer.add_scalar('steps_of_each_episode',
                          ep_steps[-1], i_episode)
    writer=pd.ExcelWriter('trajectory_Multi_energy.xlsx')
    df1=pd.DataFrame(data={'x_val':x_val,'y_val': y_val,'energy':energy})
    df1.to_excel(writer,'sheet1')
    writer.save()
    plt.plot(x_val,y_val)
    plt.plot(x_val,y_val,"^")
    plt.show()  
    plt.plot(energy)
    plt.plot(energy,"-")
    plt.show() 
    writer.close()
    '''存储reward等相关结果'''
    if not os.path.exists(RESULT_PATH): # 检测是否存在文件夹
        os.mkdir(RESULT_PATH)
    np.save(RESULT_PATH+'rewards_eval.npy', rewards)
    np.save(RESULT_PATH+'moving_average_rewards_eval.npy', moving_average_rewards)
    np.save(RESULT_PATH+'steps_eval.npy', ep_steps)

if __name__ == "__main__":
    cfg = get_args()
    if cfg.train:
        
        begin_time = time()
        train(cfg)
        eval(cfg)
    else:
        model_path = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"
        eval(cfg,saved_model_path=model_path)
