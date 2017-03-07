# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:11:02 2016

@author: mb540
"""

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

plt.axis([0, 10, 0, 1])
plt.ion()

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

#while True:
#    plt.pause(0.05)