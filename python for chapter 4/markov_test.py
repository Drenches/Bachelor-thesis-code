# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:04:37 2020

@author: dell
"""
import random as rm

N_STATES = 3
x = [[0], [200], [500]] 

transitionStates = [[0], [200], [500], [1000]]
transitionMatrix = [[0.4, 0.3, 0.2, 0.1],
                         [0.3, 0.4, 0.1, 0.2],
                         [0.2, 0.1, 0.4, 0.3],
                         [0.1, 0.2, 0.3, 0.4]]
new_env = [0] * N_STATES
for i in range(N_STATES):             
    idx = transitionStates.index(x[i])
    prVector = transitionMatrix[idx]
    pr = rm.random()
    prMap = [abs(n - pr) for n in prVector]
    new_idx = prMap.index(min(prMap))
    new_env[i] = transitionStates[new_idx]