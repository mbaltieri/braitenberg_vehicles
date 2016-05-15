# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:57:39 2016

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 1000
iterations = int(T/dt)

### agent ###
radius = 4
sensors_angle = np.pi/3                     # angle between sensor and central body line
length_dir = 3

pos = np.zeros((2,1))                       # centre of mass
vel = np.zeros((2,1))
theta = 0

# sensors
sensor = np.zeros((2,1))

# network (brain)
temp_orders = 2
nodes = 3

x = np.zeros((iterations,nodes,temp_orders))
x_init = np.random.standard_normal(nodes,)

n = .0*np.random.standard_normal((iterations,nodes))
n = np.zeros((iterations,nodes))
w_orig = np.random.standard_normal((nodes,nodes))

# data (history)
pos_history = np.zeros((2,iterations))
vel_history = np.zeros((2,iterations))
theta_history = np.zeros((1,iterations))
orientation_history = np.zeros((2,2,iterations))
sensor_history = np.zeros((2,iterations))

### environment ###

# light source
pos_light = np.array([19,27])
light_intensity = 100

def light_level(point):
    distance = np.linalg.norm(pos_light - point)
    return light_intensity/(distance**2)


### plot ###

# plot initial position
plt.close('all')
fig = plt.figure(0)
    
plt.plot(pos_light[0], pos_light[1], color='orange', marker='o', markersize=20)

orientation_endpoint = pos + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
orientation = np.concatenate((pos,orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation

plt.xlim((0,100))
plt.ylim((0,100))

# update the plot thrpugh objects
ax = fig.add_subplot(111)
line1, = ax.plot(pos[0], pos[1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
line2, = ax.plot(orientation[0,:], orientation[1,:], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma


### initialise variables ###
pos = np.array([[2.],[3.]])

omega = 0
theta = 0

x[0,:,0] = x_init
w_orig = np.array([[ 1.12538509, -2.00524372, 0.64383674], [-0.61054784, 0.15221595, -0.36371622], [-0.02720039, 1.39925152, 0.84412855]])
alpha = 1*np.ones((nodes,))

for i in range(iterations-1):
    # brain
    x[i,:,1] = 1/alpha*(- x[i,:,0] + np.tanh(np.dot(w_orig,x[i,:,0]))) + n[i,]
    x[i+1,:,0] = x[i,:,0] + dt*x[i,:,1]    
    
    # perception
    sensor[0] = light_level(pos + radius*(np.array([[np.cos(theta+sensors_angle)], [np.sin(theta+sensors_angle)]])))            # left sensor
    sensor[1] = light_level(pos + radius*(np.array([[np.cos(theta-sensors_angle)], [np.sin(theta-sensors_angle)]])))            # right sensor
    
    # action
#    vel[0] = x[i,0,1]                   # attach neuron to motor
#    vel[1] = x[i,1,1]                   # attach neuron to motor
    
    vel[0] = np.tanh(sensor[0])                   # attach neuron to motor
    vel[1] = np.tanh(sensor[1])                   # attach neuron to motor
    
    # translation
    vel_centre = (vel[0]+vel[1])/2
    pos += dt*vel_centre
    
    # rotation
    omega = np.float((vel[0]-vel[1])/(2*radius))
    theta += dt*omega
    
    # update plot
    if np.mod(i,100)==0:                                                                    # don't update at each time step, too computationally expensive
        orientation_endpoint = pos + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
        orientation = np.concatenate((pos,orientation_endpoint), axis=1)
        line1.set_xdata(pos[0])    
        line1.set_ydata(pos[1])
        line2.set_xdata(orientation[0,:])
        line2.set_ydata(orientation[1,:])
        fig.canvas.draw()
    #input("\nPress Enter to continue.")                                                    # adds a pause

    # save data    
    pos_history[:,i] = pos[:,0]
    vel_history[:,i] = vel[:,0]
    theta_history[:,i] = theta    
    #orientation_history[:,:,i] = orientation
    sensor_history[:,i] = sensor[:,0]

