# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:57:39 2016

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 100
iterations = int(T/dt)


# network
temp_orders = 2
nodes = 3

x = np.zeros((iterations,nodes,temp_orders))
y = np.copy(x);

x_init = np.random.standard_normal(nodes,)
x[0,:,0] = x_init
y[0,:,0] = x[0,:,0]

n = .0*np.random.standard_normal((iterations,nodes))
n = np.zeros((iterations,nodes))
w_orig = np.random.standard_normal((nodes,nodes))


# agent
radius = 4
angle = 60
length_dir = 3

pos = np.zeros((2,1))
vel = np.zeros((2,1))
pos = np.array([[2.],[3.]])
orien = 0

# light source
pos_light = np.array([9,7])

plt.close('all')
fig = plt.figure(0)
    
plt.plot(pos_light[0], pos_light[1], color='orange', marker='o', markersize=20)

#plt.plot(pos[0], pos[1], color='lightblue', marker='.', markersize=30*radius)
ax = fig.add_subplot(111)
line1, = ax.plot(pos[0], pos[1], color='lightblue', marker='.', markersize=30*radius) # Returns a tuple of line objects, thus the comma

aa = pos + length_dir*(np.array([[np.cos(orien+np.pi/2)], [np.sin(orien+np.pi/2)]]))
orientation = np.concatenate((pos,aa), axis=1)
#plt.plot(orientation[0,:], orientation[1,:], color='black', linewidth=2)
line2, = ax.plot(orientation[0,:], orientation[1,:], color='black', linewidth=2) # Returns a tuple of line objects, thus the comma
plt.xlim((0,20))
plt.ylim((0,20))

omega = 0
orien = 0

pos_history = np.zeros((2,iterations))
orien_history = np.zeros((2,iterations))
orientation_history = np.zeros((2,2,iterations))

w_orig = np.array([[ 1.12538509, -2.00524372, 0.64383674], [-0.61054784, 0.15221595, -0.36371622], [-0.02720039, 1.39925152, 0.84412855]])

alpha = 1*np.ones((nodes,))

for i in range(iterations-1):
    
    x[i,:,1] = 1/alpha*(- x[i,:,0] + np.tanh(np.dot(w_orig,x[i,:,0]))) + n[i,]
    x[i+1,:,0] = x[i,:,0] + dt*x[i,:,1]    
    
    
    vel[0] = x[i,0,1]
    vel[1] = x[i,1,1]
    vel_centre = (vel[0]-vel[1])/2
    pos += dt*vel_centre
    
    pos_history[:,i] = pos[:,0]
    omega = np.float((vel[0]-vel[1])/(2*radius))
    orien += dt*omega
    
    orien_history[:,i] = orien
    
    bb = length_dir*(np.array([[np.cos(orien+np.pi/2)], [np.sin(orien+np.pi/2)]]))
    aa = pos + length_dir*(np.array([[np.cos(orien+np.pi/2)], [np.sin(orien+np.pi/2)]]))
    orientation = np.concatenate((pos,aa), axis=1)
    
    orientation_history[:,:,i] = orientation 
    #
    
    # You probably won't need this if you're embedding things in a tkinter plot...
    #plt.ion()
    
    #fig = plt.figure()
    
    
    #for phase in np.linspace(0, 10*np.pi, 500):
    line1.set_xdata(pos[0])    
    line1.set_ydata(pos[1])
    line2.set_xdata(orientation[0,:])
    line2.set_ydata(orientation[1,:])
    if np.mod(i,100)==0:
        fig.canvas.draw()
    #input("\nPress Enter to continue.")
