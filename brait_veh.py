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
radius = 2
sensors_angle = np.pi/3                     # angle between sensor and central body line
length_dir = 3

pos_centre = np.zeros((2,1))                       # centre of mass
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
pos_centre_history = np.zeros((2,iterations))
vel_centre_history = np.zeros((1,iterations))
vel_history = np.zeros((2,iterations))
theta_history = np.zeros((1,iterations))
orientation_history = np.zeros((2,2,iterations))
sensor_history = np.zeros((2,iterations))

### environment ###

# light source
pos_centre_light = np.array([[39.],[47.]])
light_intensity = 200

def light_level(point):
    distance = np.linalg.norm(pos_centre_light - point)
    return light_intensity/(distance**2)
#    return light_intensity*np.exp(-distance)


### plot ###

# plot initial pos_centreition
plt.close('all')
#fig = plt.figure(0)
#    
#plt.plot(pos_centre_light[0], pos_centre_light[1], color='orange', marker='o', markersize=20)
#
#orientation_endpoint = pos_centre + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
#orientation = np.concatenate((pos_centre,orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation
#
#plt.xlim((0,100))
#plt.ylim((0,100))
#
## update the plot thrpugh objects
#ax = fig.add_subplot(111)
#line1, = ax.plot(pos_centre[0], pos_centre[1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
#line2, = ax.plot(orientation[0,:], orientation[1,:], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma


### initialise variables ###
pos_centre = np.array([[67.],[85.]])
#pos_centre = 100*np.random.random((2,1))

omega = 0
#theta = 4*np.pi/3
#theta = np.pi*2*np.random.uniform()

x[0,:,0] = x_init

noise_sens_sdv = 1
noise_sens = noise_sens_sdv*np.random.randn(2,iterations)
noise_vel = 1*np.random.randn(2,iterations)

for i in range(iterations-1):
    print(i)
    
    # perception
    sensor[0] = light_level(pos_centre + radius*(np.array([[np.cos(theta+sensors_angle)], [np.sin(theta+sensors_angle)]])))            # left sensor
    sensor[1] = light_level(pos_centre + radius*(np.array([[np.cos(theta-sensors_angle)], [np.sin(theta-sensors_angle)]])))            # right sensor
    
    sensor += noise_sens[:,i,None]
    
    # vehicle 2
#    vel[0] = np.tanh(sensor[1])                   # attach neuron to motor
#    vel[1] = np.tanh(sensor[0])                   # attach neuron to motor
    
    # vehicle 3
    vel[0] = 5*(1-1/(1+np.exp(-sensor[0])))                   # attach neuron to motor
    vel[1] = 5*(1-1/(1+np.exp(-sensor[1])))                   # attach neuron to motor
    
    vel += noise_vel[:,i,None]
    
    # translation
    vel_centre = (vel[0]+vel[1])/2
    pos_centre += dt*(vel_centre*np.array([[np.cos(theta)], [np.sin(theta)]]))
    
    # rotation
    omega = 50*np.float((vel[1]-vel[0])/(2*radius))
    theta += dt*omega
    
    # update plot
#    if np.mod(i,200)==0:                                                                    # don't update at each time step, too computationally expensive
#        orientation_endpoint = pos_centre + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
#        orientation = np.concatenate((pos_centre,orientation_endpoint), axis=1)
#        line1.set_xdata(pos_centre[0])
#        line1.set_ydata(pos_centre[1])
#        line2.set_xdata(orientation[0,:])
#        line2.set_ydata(orientation[1,:])
#        fig.canvas.draw()
    #input("\nPress Enter to continue.")                                                    # adds a pause

    # save data
    vel_centre_history[0,i] = vel_centre
    pos_centre_history[:,i] = pos_centre[:,0]
    vel_history[:,i] = vel[:,0]
    theta_history[:,i] = theta    
    #orientation_history[:,:,i] = orientation
    sensor_history[:,i] = sensor[:,0]
    
plt.figure(1)
plt.plot(pos_centre_history[0,:-1], pos_centre_history[1,:-1])
plt.plot(pos_centre_light[0], pos_centre_light[1], color='orange', marker='o', markersize=20)
plt.xlim((0,100))
plt.ylim((0,100))

plt.figure(2)
data = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        data[i,j] = light_level(np.array([i,j])) + noise_sens_sdv*np.random.randn()
plt.imshow(data, vmin=0, origin='lower')#, vmax=10)
plt.colorbar()

plt.show()

#
#
#plt.figure(3)
#plt.subplot(1,2,1)
#plt.plot(range(iterations), sensor_history[0,:], 'b')
#plt.title("Light intensity")
#plt.subplot(1,2,2)
#plt.plot(range(iterations), sensor_history[1,:], 'b')
#
#plt.figure(4)
#plt.plot(vel_history[0,:-1], vel_history[1,:-1])