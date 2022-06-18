# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 21:46:50 2022

@author: dell
"""

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
import numpy as np
import math
import gym
import random
from gym import spaces
from gym.utils import seeding
cost=0.1
#v=0.5
class NormalizedEnv():
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
        self.viewer = None
        self.step_reward=[]
        self.sp_reward=[]
        self.on_mouse = on_mouse
        self.random_goal=random_goal
        self.sensor_pos=np.array([10,10],np.float32)
        #状态最大值 包含三个，小车横坐标max_length,小车纵坐标max_width,小车内最大的油耗max_B
        high = np.array([self.max_length,self.max_width,self.max_B,360.], dtype=np.float32)
        low = np.array([0., 0., 0.,0.], dtype=np.float32)
         
        action_high=np.array([self.max_speed,self.theta_max],dtype=np.float32)
        action_low=np.array([self.min_speed,self.theta_min],dtype=np.float32)
      
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
    
    def step(self,u,t):
        
        # self.sensor_pos=np.array([random.uniform(0, 10),random.uniform(0, 10)],np.float32)
        # print("u-------------------------",u)
        u[0]=np.clip(self.action_zoom(u[0],0.,1.),0,1)     
        x,y,B,angel=self.state
        E1=random.gauss(80,1)
        E2=random.gauss(30,1)
        theta=math.degrees(math.asin(u[1]))
        self.sensor_pos[0]=10
        self.sensor_pos[1]=10
        d1=(math.sqrt((x-0)**2+(y-10)**2)+1)**2 
        d2=(math.sqrt((x)**2+(y)**2)+1)**2
        d3=(math.sqrt((x-10)**2+(y-10)**2)+1)**2
        k=random.gauss(2,1)
        reward=k*B/d3
        B=E1/d1+E2/d2 
        angel+=theta 
        angel %=360
        if B>u[0]*cost:
            x=x+u[0]*math.cos(math.radians(angel))
            y=y+u[0]*math.sin(math.radians(angel))
            B=B-u[0]*cost
            
        angel=np.clip(angel,0,360)
        x=np.clip(x,0.,10.)
        y=np.clip(y,0.,10.)
        B=np.clip(B,self.min_B,self.max_B)
        
        self.sensor_pos[0]=np.clip(self.sensor_pos[0],0.,10.)
        self.sensor_pos[1]=np.clip(self.sensor_pos[1],0.,10.)
        self.state=np.array([x,y,B,angel])
        return self.state,reward,False,{},self.sensor_pos
    def reset(self):
        self.sensor_pos=np.array([10,10],np.float32)
        self.state=np.array([5.,5.,50.,0.],dtype=np.float64)
        x_,y_,B_,xita_=self.state
        # print("初始化的状态值",self.state)
        return np.array([x_,y_,B_,xita_]),self.sensor_pos
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
    def __init__(self, goal_velocity=0):
        #self.min_action = -1.0
        #self.max_action = 1.0
        self.low_state = np.array(
            [0., 0.,0.,0.], dtype=np.float32
        )
        self.high_state = np.array(
            [10., 10.,360.,50.], dtype=np.float32
        )
        self.low_action = np.array(
            [-1., 0.], dtype=np.float32
        )
        self.high_action = np.array(
            [1., 1.], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.reset()
    def angle_normalize(self,x):
        return ((x+180) % 360) - 180
    def v_normalize(self,x):
        return 0.5*(x+1)
    def step(self, u):
        x,y,the,B=self.state
        #print("初始状态：",x,y,the,B)
        E1=random.gauss(80,1)#[0,10]
        E2=random.gauss(30,1)#[0,0]
        d1=(math.sqrt(x**2+(y-10)**2)+1)**2
        d2=(math.sqrt(x**2+y**2)+1)**2
        d3=(math.sqrt((x-10.)**2+(y-10.)**2)+1)**2 #sensor的位置[10,10]
        p=random.gauss(1,1)#[0,0]
        reward=p*B/d3
        B=E1/d1+E2/d2
        #u = np.clip(u,-1,1)
        the+=math.degrees(math.asin(u[0]))
        v=self.v_normalize(u[1])
        #print("math.asin(u)，%%%the%%%",math.asin(u),the)
        the =self.angle_normalize(the)
        #print("归一化后的角度和速度：",the,v)
        if B>v*cost:
           B=B-v*cost
           x=x+v*math.cos(math.radians(the))
           y=y+v*math.sin(math.radians(the))
        x=np.clip(x,0.,10.)
        y=np.clip(y,0.,10.)
        # 表示传感器正常工作需要的能量，是一个随机变量
        #E_sensor_random=random.gauss(50,0)
        self.state=np.array([x,y,the,B])
        return self.state,reward,False,{}
    def reset(self):
        self.state=np.array([5.,5.,0.,50.], dtype=np.float32)
        return self.state
"""