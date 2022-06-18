#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:28:30
@LastEditor: John
LastEditTime: 2020-09-01 10:57:36
@Discription: 
@Environment: python 3.7.7
'''

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random
import math
import torch
# import main
# main.train()

cost=0.1
class PendulumEnv():
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,on_mouse=False,random_goal=False):
        self.max_speed=1.
        self.min_speed=0.
        # self.speed=0.2
        self.max_length=10.
        self.min_length=0.
        self.max_width=10.
        self.min_width=0.
        self.max_B=10000.
        self.min_B=0.
        self.theta_max=1.0
        self.theta_min=-1.0
        # self.min_angel=0.
        # self.max_angel=360
        # angel=0.
        # self.Ep_max=1000.
        # self.Ep_min=100
        # self.max_E_sensor=1000
        # self.E_sensor=10
        self.viewer = None
        self.step_reward=[]
        self.sp_reward=[]
        self.on_mouse = on_mouse
        self.random_goal=random_goal
        self.sensor_pos=np.array([5,10],np.float32)
        #状态最大值 包含三个，小车横坐标max_length,小车纵坐标max_width,小车内最大的油耗max_B
        high = np.array([self.max_length,self.max_width,self.max_B,360.], dtype=np.float32)
        low = np.array([0., 0., 0.,0.], dtype=np.float32)
         
        action_high=np.array([self.max_speed,self.theta_max],dtype=np.float32)
        action_low=np.array([self.min_speed,self.theta_min],dtype=np.float32)
        # action_low=self.theta_min
        # action_high=self.theta_max
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            # shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
       
        print("动作空间：",self.action_space)
        print("状态空间：",self.observation_space)
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def action_zoom(self,action,min,max):
        m=0.5*(action+1)*(max-min)+min
        return m
    
    def step(self, u,t):
        
        # self.sensor_pos=np.array([random.uniform(0, 10),random.uniform(0, 10)],np.float32)
        # print("u-------------------------",u)
        u[0]=np.clip(self.action_zoom(u[0],0.,1.),0,1)     
        x,y,B,angel=self.state
        E1=random.gauss(80,1)
        E2=random.gauss(30,1)
        E3=random.gauss(50,1)
        E4=random.gauss(100,1)
        theta=math.degrees(math.asin(u[1]))
        self.sensor_pos[0]=5
        self.sensor_pos[1]=10
        #d1:energy1(0,10),d2:energy2(0,0),d3:energy3(6,8),d4:energy4(2,2)| d:sink(10,10)
        d1=(np.sqrt((x-5)**2+(y-5)**2)+1)**2
        d2=(np.sqrt((x-4)**2+(y-6)**2)+1)**2
        d3=(np.sqrt((x-3)**2+(y-3)**2)+1)**2
        d4=(np.sqrt((x-6)**2+(y-5)**2)+1)**2
        d5=(np.sqrt((x-5)**2+(y-4)**2)+1)**2
        d6=(np.sqrt((x-4)**2+(y-5)**2)+1)**2
        d=(np.sqrt((x-5)**2+(y-10)**2)+1)**2
        # print("d3--------",d3)
        # print("self.sensor_pos--------",self.sensor_pos[0],self.sensor_pos[1])
        #k=random.gauss(2,1)
        reward=B/d
        # print("reward!!!!!!!!!!!!!!!!!!!!!!!",reward)
        B=(100/d1+80/d2+50/d3+2200/d4+100/d5+80/d6)/d
        # print("原来的角度-----------",B)
        # print("原来的角度-----------",angel)
        angel+=theta 
        # print("动作之后的角度-------------",angel)
        angel %=360
        # print("角度归一化----------------",angel)
        
        # print("移动之前  的 x +++++++++",x)
        # print("移动之前  的 y ++++++++",y)
        # print("移动之前  的小车能量 ++++++++++++++++",B)
        if B>u[0]*cost:
            x=x+u[0]*math.cos(math.radians(angel))
            y=y+u[0]*math.sin(math.radians(angel))
            B=B-u[0]*cost
            # print("移动之后的x+++++++++",x)
            # print("移动之后的y ++++++++",y)
            # print("移动之后的小车能量 ++++++++++++++++",B)
            
        angel=np.clip(angel,0,360)
        x=np.clip(x,0.,10.)
        y=np.clip(y,0.,10.)
        # print("x","y","B",x,y,B)
        B=np.clip(B,self.min_B,self.max_B)
        self.sensor_pos[0]=np.clip(self.sensor_pos[0],0.,10.)
        self.sensor_pos[1]=np.clip(self.sensor_pos[1],0.,10.)
        # print("**********************",self.sensor_pos)
        # print(" self.sensor_pos[0]", self.sensor_pos[0])
        # print(" self.sensor_pos[1]", self.sensor_pos[1])
        self.state=np.array([x,y,B,angel])
        return self.state,reward,False,{},self.sensor_pos
    def reset(self):
        # high = np.array([self.max_length,self.max_width,10.,0.], dtype=np.float32)
        # low=np.array([0.,0.,10.,0.],dtype=np.float32)
        # self.state=self.np_random.uniform(low=low,high=high)
      
        # if not self.on_mouse:
        # if self.random_goal:
        #     r = np.pi * 2 * np.random.rand()
        #     d = self.max_width / 2 * np.random.rand()
        #     self.goal_pos[0] = np.cos(r) * d
        #     self.goal_pos[1] = np.sin(r) * d
        #         # self.goal_pos += self.arm_pos_shift
        # else:
        #         # sensor pos不变
        #     self.sensor_pos=[10.,10.]
        # ,self.sensor_pos[0],self.sensor_pos[1]
        
        self.sensor_pos=np.array([5,10],np.float32)
        # self.sensor_pos=np.array([random.uniform(0, 10),random.uniform(0, 10)],dtype=np.float32) 
        # self.sensor_pos=np.array([10.,0.],np.float32)
        self.state=np.array([10.,5.,50.,0.],dtype=np.float64)
        x_,y_,B_,xita_=self.state
        # print("初始化的状态值",self.state)
        return np.array([x_,y_,B_,xita_]),self.sensor_pos
    
