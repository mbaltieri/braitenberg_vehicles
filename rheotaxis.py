# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:36:29 2016

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 1
iterations = int(T/dt)


### agent ###
radius = 4
sensors_n = 2
motors_n = 2
sensors_angle = np.pi/3
length_dir = 3
pos_centre = np.array([[20.],[50.]])
vel = np.zeros((motors_n,1))
acc = np.zeros((motors_n,1))
theta = np.pi
sensor = np.zeros((sensors_n,))

# data (history)
pos_centre_history = np.zeros((2,iterations))
vel_centre_history = np.zeros((1,iterations))
sensor_history = np.zeros((sensors_n,iterations))


def pressure(point):
    return np.exp(-point[1,0])



plt.close('all')

fig = plt.figure(0)

plt.axvline(x=-20, ymin=0, ymax=100)

orientation_endpoint = pos_centre + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
orientation = np.concatenate((pos_centre,orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation

# update the plot through objects
ax = fig.add_subplot(111)
line1, = ax.plot(pos_centre[0], pos_centre[1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
line2, = ax.plot(orientation[0,:], orientation[1,:], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma


plt.xlim((-50,50))
plt.ylim((0,100))
plt.show()

for i in range(iterations):
    print(i)
    
    # perception
    sensor[0] = pressure(pos_centre + radius*(np.array([[np.cos(theta+sensors_angle)], [np.sin(theta+sensors_angle)]])))            # left sensor
    sensor[1] = pressure(pos_centre + radius*(np.array([[np.cos(theta-sensors_angle)], [np.sin(theta-sensors_angle)]])))            # right sensor
    
    
    acc[0] = -sensor[0]
    acc[1] = -sensor[1]
    
    vel += dt*acc
    
    # translation
    vel_centre = (vel[0]+vel[1])/2
    pos_centre += dt*(vel_centre*np.array([[np.cos(theta)], [np.sin(theta)]]))
    
        
    # update plot
#    if np.mod(i,200)==0:                                                                    # don't update at each time step, too computationally expensive
    orientation_endpoint = pos_centre + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
    orientation = np.concatenate((pos_centre,orientation_endpoint), axis=1)
    line1.set_xdata(pos_centre[0])
    line1.set_ydata(pos_centre[1])
    line2.set_xdata(orientation[0,:])
    line2.set_ydata(orientation[1,:])
    fig.canvas.draw()
    #input("\nPress Enter to continue.") 


    # history

    vel_centre_history[0,i] = vel_centre
    pos_centre_history[:,i] = pos_centre[:,0]
    sensor_history[:,i] = sensor[:]
    
    
    
    
    
    
    
    
    