# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:03:53 2020

@author: dell
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""不同learning rate之间对比"""
# x = np.load("lr_01.npy")
# y = np.load("lr_001.npy")
# z = np.load("lr_0001.npy")

# x = x[0:700]
# y = y[0:700]
# z = z[0:700]

# plt.plot(x, color='blue', label='lr = 0.1')
# plt.plot(y, color='orange', label='lr = 0.01')
# plt.plot(z, color='green', label='lr = 0.001')

# plt.legend()
# plt.grid()
# plt.ylabel(r'Moving average reward')
# plt.xlabel(r'step $\qquad \times 50$')
# plt.show()

"""不同batch_size下Loss之间对比"""
# x = np.load("loss_8.npy")[1:10000:10]
# y = np.load("loss_32.npy")[1:10000:10]
# z = np.load("loss_64.npy")[1:10000:10]
# m = np.load("loss_128.npy")[1:10000:10]

# plt.plot(x, color='blue', label='batch_size = 8')
# plt.plot(y, color='orange', label='batch_size = 32')
# plt.plot(z, color='green', label='batch_size = 64')
# plt.plot(m, color='red', label='batch_size = 128')

# plt.legend()
# plt.grid()
# plt.ylabel(r'Loss')
# plt.xlabel(r'step $\qquad \times 10$')
# plt.show()

"""不同搜索策略之间对比"""
# x = np.load("lr_001.npy")[0:400]
# y = np.load("MAR_random.npy")[0:400]
# z = np.load("MAR_naive.npy")[0:400]

# plt.plot(x, color='blue', label='greedy')
# plt.plot(y, color='orange', label='random')
# plt.plot(z, color='green', label='naive')

# plt.legend()
# plt.grid()
# plt.ylabel(r'Moving average reward')
# plt.xlabel(r'step $\qquad \times 50$')
# plt.show()

"""其他方法比较"""
x = [0.066, 0.054, 0.048, 0.039, 0.032, 0.027, 0.026]
m = [0.055, 0.049, 0.042, 0.030, 0.021, 0.021, 0.015]
n = [0.016, 0.015, 0.013, 0.010, 0.009, 0.0072, 0.0065]
y = [5, 7, 10, 12, 15, 17, 20]

x = np.load("PA.npy")
y = np.load("PFA.npy")
n = np.load("Rdm.npy")

y = [5, 7, 10, 12, 15, 17, 20]

plt.plot(y, x, marker='o', color='blue', label='Proposed approach')
plt.plot(y, m, marker='v', color='orange', label='PFA')
plt.plot(y, n, marker='d', color='green', label='Random')

plt.legend()
plt.grid()
plt.ylabel(r'Final Moving average reward')
plt.xlabel(r'Number of users')
plt.show()

