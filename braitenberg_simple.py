#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:34:30 2017

Standard implementation of a Braitenberg vehicle

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.fftpack

dt_brain = .005
dt_world = .0005
T = 200
iterations = int(T/dt_brain)
plt.close('all')
np.random.seed(42)

sensors_n = 2
motors_n = 2
obs_states = sensors_n
hidden_states = obs_states                                  # x, in Friston's work
hidden_causes = sensors_n                                   # v, in Friston's work
states = obs_states + hidden_states

### Braitenberg vehicle variables
radius = 2
sensors_angle = np.pi/3                                     # angle between sensor and central body line
length_dir = 3                                              # used to plot?
max_speed = 100.

l_max = 200.
turning_speed = 30.

### Global functions ###

def light_level(x_agent):
    x_light = np.array([59.,47.])
    sigma_x = 30.
    sigma_y = 30.
    Sigma = np.array([[sigma_x ** 2, 0.], [0., sigma_y ** 2]])
    mu = x_light
    corr = Sigma[0, 1] / (sigma_x * sigma_y)
    
    return 5655 * l_max / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - corr ** 2)) * np.exp(
            - 1 / (2 * (1 - corr ** 2)) * ((x_agent[0] - mu[0]) ** 2 / 
            (sigma_x ** 2) + (x_agent[1] - mu[1]) ** 2 / (sigma_y ** 2) - 
            2 * corr * (x_agent[0] - mu[0]) * (x_agent[1] - mu[1]) / (sigma_x * sigma_y)))


# free energy functions
def g(x, v):
    return x

def f(x_agent, v_agent, v_motor, theta, v, w, a, i):
#    # vehicle 3a - lover
#    v_motor[i, 0] = l_max - a[0]
#    v_motor[i, 1] = l_max - a[1]
    
    # vehicle 2b - aggressor
    v_motor[i, 0] = a[1]
    v_motor[i, 1] = a[0]
    
    # translation
    v_agent[i] = (v_motor[i, 0] + v_motor[i, 1]) / 2
    x_agent[i + 1, :] = x_agent[i, :] + dt_world * (v_agent[i] * np.array([np.cos(theta[i]), np.sin(theta[i])]))
        
    # rotation
    omega = turning_speed * np.float((v_motor[i, 1] - v_motor[i, 0]) / (2 * radius))
    theta[i + 1] = theta[i] + dt_world * omega
    theta[i + 1] = np.mod(theta[i + 1], 2 * np.pi)
    
    # return level of light for each sensor
    
    sensor = np.zeros(2, )
    
    sensor[0] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    sensor[1] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    return sensor, v_motor[i, :]

def getObservation(x_agent, v_agent, v_motor, theta, v, w, z, a, iteration):
    x, v_motor = f(x_agent, v_agent, v_motor, theta, v, w, a, iteration)
    return (g(x, v), g(x, v) + z)

def Braitenberg(noise_level, desired_confidence, z2):
    x_light = np.array([59.,47.])
    
    # noise on sensory input
    gamma_z = noise_level * np.ones((obs_states, ))    # log-precisions
    pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
    sigma_z = 1 / (np.sqrt(pi_z))
    z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()
    z = z2
    
    # noise on motion of hidden states
    gamma_w = desired_confidence * np.ones((hidden_states, ))    # log-precision
    pi_w = np.exp(gamma_w) * np.ones((hidden_states, ))
    sigma_w = 1 / (np.sqrt(pi_w))
    w = (np.dot(np.diag(sigma_w), np.random.randn(obs_states, iterations))).transpose()
    
    s2 = np.zeros((iterations, sensors_n))
    theta2 = np.zeros((iterations, ))                            # orientation of the agent
    x_agent2 = np.zeros((iterations, 2))                         # 2D world, 2 coordinates por agent position
    v_agent2 = np.zeros((iterations, ))
    v_motor2 = np.zeros((iterations, motors_n))
    rho2 = np.zeros((iterations, obs_states))
    filtered_rho2 = np.zeros((iterations, obs_states))
    
    ### initialisation
    
#    x_agent2[0, :] = np.array([10., 10. * np.random.rand()])
    x_agent2[0, :] = np.array([0., 0.])
    #x_agent2[0, :] = 100 * np.random.rand(1, 2)
    
    #theta[0] = np.pi * np.random.rand()
    theta2[0] = np.pi / 2 #2 / 3 * np.pi
    
    ## online plot routine
#    fig = plt.figure(10)
#    plt.ion()
#        
#    #plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#    
#    orientation_endpoint = x_agent2[0, :, None] + length_dir * (np.array([[np.cos(theta2[0])], [np.sin(theta2[0])]]))
#    orientation = np.concatenate((x_agent2[0, :, None], orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation
#    
#    plt.xlim((0,100))
#    plt.ylim((0,200))
#    
#    # update the plot through objects
#    ax = fig.add_subplot(111)
#    line1, = ax.plot(x_agent2[0, 0], x_agent2[0, 1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
#    line2, = ax.plot(orientation[0, :], orientation[1, :], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma

    decay = 5
    
    s2[0, :], rho2[0, :] = getObservation(x_agent2, v_agent2, v_motor2, theta2, 0., w[0, :], z[0, :], (s2[0, :] + z[0, :] / np.sqrt(dt_brain)), 0)
    for i in range(1, iterations - 1):
        s2[i, :], rho2[i, :] = getObservation(x_agent2, v_agent2, v_motor2, theta2, 0., w[i, :], z[i, :], s2[i - 1, :] + z[i - 1, :] / np.sqrt(dt_brain), i)
#        s2[i, :], rho2[i, :] = getObservation(x_agent2, v_agent2, v_motor2, theta2, 0., w[i, :], z[i, :], filtered_rho2[i - 1, :], i)
#        s2[i, :], rho2[i, :] = getObservation(x_agent2, v_agent2, v_motor2, theta2, 0., w[i, :], z[i, :], rho2[i - 1, :], i)
        
        filtered_rho2[i, :] = filtered_rho2[i - 1, :] + dt_brain * decay * (s2[i, :] + z[i, :] / np.sqrt(dt_brain) - filtered_rho2[i - 1, :])
#        orientation_endpoint = x_agent2[i, :, None] + length_dir * (np.array([[np.cos(theta2[i])], [np.sin(theta2[i])]]))
#        orientation = np.concatenate((x_agent2[i, :, None], orientation_endpoint), axis=1)
#        line1.set_xdata(x_agent2[i, 0])
#        line1.set_ydata(x_agent2[i, 1])
#        line2.set_xdata(orientation[0,:])
#        line2.set_ydata(orientation[1,:])
#        fig.canvas.draw()
#        plt.pause(0.05)
    
#    plt.figure()
#    plt.plot(x_agent2[:, 0], x_agent2[:, 1])
#    plt.xlim((0,80))
#    plt.ylim((0,80))
#    plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#    plt.plot(x_agent2[0, 0], x_agent2[0, 1], color='red', marker='o', markersize=8)
#    plt.title('Trajectory (Braitenberg)')
    
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(x_agent2[:-1, 0], 'b', np.ones(iterations - 1) * x_light[0], 'r')
#    plt.subplot(1, 2, 2)
#    plt.plot(x_agent2[:-1, 1], 'b', np.ones(iterations - 1) * x_light[1], 'r')
#    plt.title('Position')
#        
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(rho2[:-1, 0], 'b', label='Noisy signal')
#    plt.plot(s2[:-1, 0], 'g', label='Signal')
#    plt.legend()
#    plt.subplot(1, 2, 2)
#    plt.plot(rho2[:-1, 1], 'b', label='Noisy signal')
#    plt.plot(s2[:-1, 1], 'g', label='Signal')
#    plt.legend()
#    
#    plt.figure()
#    plt.plot(v_motor2[:-1, 0], 'b', label='Motor1')
#    plt.plot(v_motor2[:-1, 1], 'r', label='Motor2')
#    plt.title('Velocity')
#    plt.legend()
    #
    #plt.figure()
    #plt.plot(theta2)
    #plt.title('Orientation')
    #
#    points = 100
#    x_map = range(points)
#    y_map = range(points)
#    light = np.zeros((points, points))
#    
#    for i in range(points):
#        for j in range(points):
#            light[i, j] = light_level(np.array([x_map[j], y_map[i]])) + sigma_z[0] * np.random.randn()
#    
#    light_fig = plt.fianagure()
#    light_map = plt.imshow(light, extent=(0., points, 0., points),
#               interpolation='nearest', cmap='jet')
#    cbar = light_fig.colorbar(light_map, shrink=0.5, aspect=5)
#    
    return x_agent2, s2, rho2, filtered_rho2



noise_level = - 3.
gamma_z = noise_level * np.ones((obs_states, ))    # log-precisions
pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
real_pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
sigma_z = 1 / (np.sqrt(real_pi_z))
z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()


agent_position, s, rho, filtered_rho = Braitenberg(noise_level, 0, z)

x_light = np.array([59.,47.])


plt.figure(figsize=(5, 4))
plt.plot(agent_position[:, 0], agent_position[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
plt.title('Trajectory', fontsize=14)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt_brain, dt_brain), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
plt.plot(np.arange(0, T-dt_brain, dt_brain), s[:-1, 0], 'g', label='Sensory reading $ρ_{l_1}$, no noise')
#plt.plot(np.arange(0, T-dt_brain, dt_brain), filtered_rho[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance')
#plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)

#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_x[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
#plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Luminance, Motor velocity')
#plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
#plt.legend(loc = 4)
#
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T-dt_brain, dt_brain), rho_m[:-1, 1], 'b', label='Motor reading $ρ_{m_2}$')
#plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Velocity')
#plt.title('Proprioceptor $ρ_{m_2}$, $\mu_{m_2}$', fontsize=14)
#plt.legend(loc = 4)


