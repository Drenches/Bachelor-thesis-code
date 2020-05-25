# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:20:06 2020

@author: dell
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.ma as ma
import random as rm
import matplotlib.pyplot as plt
from random import choice

#Initialization paramter(learnig system)
E_max = 10
H_max = 10000
U = 32
gamma = 0.75
LR = 0.01
epsilon = 0.9
replay_buffer = 2000
TARGET_REPLACE_ITER = 100

#Initialization paramter(MEC system)
alaph = 10.00
beta = 1.00
theta = 1.00
N_STATES = 10
ACTION = [100, 500, 1000, 2000, 4000, 5000]
N_ACTIONS = len(ACTION)

#Building environment
class ENV(object):
    def __init__(self):
        self.transitionStates = [0, 200, 500, 1000]
        self.transitionMatrix = [[0.4, 0.3, 0.2, 0.1],
                                 [0.3, 0.4, 0.1, 0.2],
                                 [0.2, 0.1, 0.4, 0.3],
                                 [0.1, 0.2, 0.3, 0.4]]
    def reset(self):
        idxs = np.random.randint(0, len(self.transitionStates), size=N_STATES) 
        re_env = [self.transitionStates[i] for i in idxs]
        return re_env
        
    def update(self, x): # 有限马尔科夫过程
        new_env = [0] * N_STATES
        for i in range(N_STATES):             
            idx = self.transitionStates.index(x[i])
            prVector = self.transitionMatrix[idx]
            pr = rm.random()
            prMap = [abs(n - pr) for n in prVector]
            new_idx = prMap.index(min(prMap))
            new_env[i] = self.transitionStates[new_idx]
        return new_env

    def feedback(self, s, a_aciton):
        F_mis = 100.00#clients生成一个任务签名的算力
        w = 0.10#平均transaction大小
        f = 2/3 * N_STATES
        y = s
        minval = np.array(y)
        Op = np.min(ma.masked_where(minval == 0, minval)) #求最大传输,即最小非0算力
        # clients将任务签名并广播（transaction#2）
        delta1 = beta
        T1 = delta1/F_mis
        delta2 = beta + theta + 0.95*alaph
        T2 = delta2/Op
        # 执行后将结果返回给client，client验证（transaction#2）
        delta3 = theta
        T3 = delta3/F_mis
        # 开始一个PBFT过程，client发出request, 但不需要reply
        delta4 = (a_action * (beta + theta)) / w
        T4 = delta4/Op
        delta5 = beta + theta + a_action/w * (beta +theta) + 3/4*f*alaph
        T5 = delta5/Op
        delta6 = beta + (N_STATES - 1)*theta + f*(theta + beta)
        T6 = delta6/Op
        delta7 = beta + (N_STATES - 1)*theta + f*(theta + beta)
        T7 = delta7/Op
        Tc = T1 + T2 + T3 + T4 + T5 + T6 + T7
        return Tc

#Initialiazaiton network
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((replay_buffer, N_STATES * 2 + 2))       # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.FloatTensor(x)
        # input only one sample
        #if np.random.uniform() < epsilon:   # greedy
        actions_value = self.eval_net.forward(x)
        test = actions_value.detach().numpy()
        action = np.where(test == test[1])[0][0]
        # import pdb;pdb.set_trace()
            #action = np.where(test == test.max())[0][0]
        #else:   # random
        # action = np.random.randint(0, N_ACTIONS) #这里得到的是动作的索引而不是动作本身
        return action
    
    def long_term_reward_get(self, s_, r):
        s_tensor = torch.Tensor(s_)
        future_reward = dqn.eval_net.forward(s_tensor)
        future_reward_data = future_reward.detach().numpy()
        long_term_reward_data = r + future_reward_data.max()
        return long_term_reward_data
    
    def store_transition(self, s, a_idx, r, s_):
        # import pdb;pdb.set_trace()
        transition = np.hstack((s, [a_idx, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % replay_buffer
        self.memory[index, :] = transition
        self.memory_counter += 1
        
        # if np.isnan(self.memory.sum()):
        #     import pdb;pdb.set_trace()
    
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval_net参数传给target_net
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(replay_buffer, U)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        
        
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)
        q_eval = torch.gather(q_eval, 1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(U, 1)   # 原始方法
        # q_next_list = q_next.tolist()
        # f_r = [0]*len(q_next_list)
        # for i in range(len(q_next_list)):
        #     # f_r[i] = choice(q_next_list[i]) #random方法
        #     f_r[i] = q_next_list[i][0] #
        # f_r = torch.Tensor(f_r).view(U, 1)
        # q_target = b_r + gamma * f_r
        loss = self.loss_func(q_eval, q_target)
        #import pdb;pdb.set_trace()

        self.optimizer.zero_grad()
        loss.backward()
        l2norm = nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10, norm_type=2)
        #nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        
        return loss      
      
dqn = DQN()
env = ENV()
ep_r = 0
term_reward = np.zeros(1)
long_term_reward = np.zeros(1)
moving_ave_eq_re = np.zeros(1)
loss_sqe = np.zeros(1)
moving_counter = 0
for i_episode in range(E_max):
    s = env.reset()
    for t in range(H_max):
        a_idx = dqn.choose_action(s)
        a_action = ACTION[a_idx]             #选择动作，即区块大小
        tc = env.feedback(s, a_action)       #执行动作
        r = 1/tc
        term_reward = np.append(term_reward, r) #收集每一步的reward
        moving_counter += 1
        if moving_counter == 50:
            moving_ave_eq_re = np.append(moving_ave_eq_re, np.mean(term_reward))#收集移动平均reward
            term_reward = np.zeros(1)
            moving_counter = 0
        while True:
            s_ = env.update(s)               #更新环境,不全为0即更新成功
            if np.array(s_).sum() == 0:
                continue
            else:
                break
        # long_term_reward_data = dqn.long_term_reward_get(s_, r)
        # long_term_reward = np.append(long_term_reward, long_term_reward_data) #收集每一步长期reward
        dqn.store_transition(s, a_idx, r, s_)
        ep_r += r
        if dqn.memory_counter > replay_buffer:
             loss_sqe = np.append(loss_sqe, dqn.learn().detach().numpy().tolist()) #训练并收集Loss
        s = s_
plt.plot(moving_ave_eq_re)
plt.show()