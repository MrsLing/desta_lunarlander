import numpy as np


class OUNoise(object):
    def __init__(self,action_space,mu=0.0,theta=0.1,max_sigma=0.2,min_sigma=0.2,decay_period=100000):
        self.mu=mu
        self.theta=theta
        self.sigma=max_sigma
        self.max_sigma=max_sigma
        self.min_sigma=min_sigma
        self.decay_period=decay_period
        self.n_actions=action_space.shape[0]
        self.low=action_space.low
        self.high=action_space.high
        self.epsilon=1.5
        self.greedy_accel_noise_steps=1000000
        self.reset()
      
    def reset(self):
        self.obs=np.ones(self.n_actions)*self.mu
    # def evolve_obs(self):
    #     x=self.obs
    #     dx=self.theta*(self.mu-x)+self.sigma*np.random.randn(self.n_actions)
    #     self.obs=x+dx
    #     return self.obs
    
    # def get_action(self,action,t=0):
    #     ou_obs=self.evolve_obs()
    #     self.sigma=self.max_sigma-(self.max_sigma-self.min_sigma)*min(1.0,t/self.decay_period)
    #     # print("action1111",action[0])
    #     # print("self.low-------",self.low)
    #     # print("self.high!!!!!!!!!!",self.high)
    #     return np.clip(action+ou_obs,self.low,self.high)  
# --------------------------------------------------------------------------------------------------------------------
    def greedy_function(self,x, mu, theta, sigma):        
        return theta * (mu - x) + sigma * np.random.randn()     
    def get_action(self,output):       
        # global epsilon 
        # print("output---",output)
        stochastic_action=[]
     
        # output = np.squeeze(output, axis=0)
        # print("output++++++++++++",output)
        self.epsilon -= 1.0 /self.greedy_accel_noise_steps 
        # print("self.epsilon*************", self.epsilon)
        greedy_noise=np.array([max(self.epsilon, 0.01) * self.greedy_function(output[0], 0.50, 0.60, 0.20),max(self.epsilon, 0.01) * self.greedy_function(output[1], 0.0, 0.10, 0.10)])      
      # greedy_noise=np.array( [max(epsilon, 0) * greedy_function(output[0], 0.0, 0.60, 0.30),  # steer
      #                 max(epsilon, 0) * greedy_function(output[1], 0.5, 1.00, 0.10),  # accel
      #                 max(epsilon, 0) * greedy_function(output[2], -0.1, 1.00, 0.05)]) # brake
        # print("greedy_noise+++++++++++++++++", greedy_noise)
        stochastic_action = np.array([greedy_noise[0] + output[0],greedy_noise[1] + output[1]])
        # print("stochastic_action……………………",stochastic_action)
        bounded =np.array([ np.clip(stochastic_action[0],-1.,1.0),np.clip(stochastic_action[1],-1.0, 1.0)])
             
        # print("bounded          ",bounded )
        return bounded

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
    # def __init__(self, action_space, mu=0.0,mu1=180,mu2=600, theta=0.15, max_sigma0=0.3, min_sigma0=0.3, max_sigma1=1,min_sigma1=1,max_sigma2=2,min_sigma2=2,decay_period=100000):
    #     self.mu = mu
    #     self.theta = theta
    #     self.sigma0 = max_sigma0
    #     self.max_sigma0 = max_sigma0
    #     self.min_sigma0 = min_sigma0
        
    #     self.mu1 = mu1
    #     self.sigma1 = max_sigma1
    #     self.max_sigma1 = max_sigma1
    #     self.min_sigma1 = min_sigma1
        
    #     self.mu2 = mu2
    #     self.sigma2 = max_sigma2
    #     self.max_sigma2 = max_sigma2
    #     self.min_sigma2 = min_sigma2
        
    #     self.decay_period = decay_period
    #     self.n_actions = action_space.shape[0]
    #     self.low = action_space.low
    #     self.high = action_space.high
    #     self.reset()
        

    # def reset(self):
    #     self.obs = np.ones(self.n_actions) * self.mu
        
    # def evolve_obs(self):
    #     self.zs=np.random.randn(self.n_actions)
        
    #     # print("self.zs",self.zs)
        
    #     x0 = self.obs[0]
    #     dx0 = self.theta * (self.mu - x0) + self.sigma0 * self.zs[0]
    #     self.obs[0] = x0+dx0
    #     # print("self.zs[0]",self.zs[0])
        
    #     x1=self.obs[1]        
    #     dx1=self.theta * (self.mu1 - x1) + self.sigma1 * self.zs[1]
    #     self.obs[1]=x1+dx1      
        
    #     x2=self.obs[2]        
    #     dx2=self.theta * (self.mu2 - x2) + self.sigma2 * self.zs[2]
    #     self.obs[2]=x2+dx2
    #     # print("...............",self.obs[0],self.obs[1],self.obs[2])
    #     return self.obs[0],self.obs[1],self.obs[2]

    # def get_action(self, action, t=0):
    #     # print("  .........",action)
    #     ou_obs0,ou_obs1,ou_obs2 = self.evolve_obs()
    #     self.sigma0 = self.max_sigma0 - (self.max_sigma0 - self.min_sigma0) * min(1.0, t / self.decay_period)
    #     self.sigma1 = self.max_sigma1 - (self.max_sigma1 - self.min_sigma1) * min(1.0, t / self.decay_period)
    #     self.sigma2 = self.max_sigma2 - (self.max_sigma2 - self.min_sigma2) * min(1.0, t / self.decay_period)
    #     action=np.array([np.clip(action[0]+ou_obs0,0,2),np.clip(action[1]+ou_obs1,0,360),np.clip(action[2]+ou_obs2, 0, 800)])
    #     # print("ou-action",action)
    #     return action