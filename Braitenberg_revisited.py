#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:12:50 2017

Second attempt of building Braitenberg vehicles based on Free Energy minimisation

In the definition of variables, hidden_states > hidden_causes and even when 
I could use hidden_causes to define smaller arrays most of the time I still use 
hidden_states to get easier matrix multiplications, the extra elements are = 0.

@author: mb540
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.fftpack

dt = .005
T = 30
iterations = int(T/dt)
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

#def sigmoid(x):
#    # vehicles 3
##    return 1 / (1 + np.exp(- 2 * x / l_max))
#    return 1 * np.tanh(x / l_max)
#
#def spinsToVelocity(x, i):
#    r = .7
##    f = 10.
#    return 2 * np.pi * r ** 2 * x / (i + 1 * dt) * 10
#
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
    
#    sigma_x = 50.
#    mu = 80.
#    return 73 * l_max / (np.sqrt(2 * sigma_x ** 2 * np.pi)) * np.exp(- (x_agent[0] - mu) ** 2 / (2 * sigma_x ** 2))


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
    x_agent[i + 1, :] = x_agent[i, :] + dt * (v_agent[i] * np.array([np.cos(theta[i]), np.sin(theta[i])]))
        
    # rotation
    omega = turning_speed * np.float((v_motor[i, 1] - v_motor[i, 0]) / (2 * radius))
    theta[i + 1] = theta[i] + dt * omega
    theta[i + 1] = np.mod(theta[i + 1], 2 * np.pi)
    
    # return level of light for each sensor
    
    sensor = np.zeros(2, )
    
    sensor[0] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    sensor[1] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    return sensor

def fFE(x_agent, v_agent, v_motor, theta, v, w, a, i):
#    # vehicle 3a - lover
#    v_motor[i, 0] = l_max - a[0]
#    v_motor[i, 1] = l_max - a[1]
    
    # vehicle 2b - aggressor
    v_motor[i, 0] = a[1]
    v_motor[i, 1] = a[0]
    
    # translation
    v_agent[i] = (v_motor[i, 0] + v_motor[i, 1]) / 2
    x_agent[i + 1, :] = x_agent[i, :] + dt / 10 * (v_agent[i] * np.array([np.cos(theta[i]), np.sin(theta[i])]))
        
    # rotation
    omega = turning_speed * np.float((v_motor[i, 1] - v_motor[i, 0]) / (2 * radius))
    theta[i + 1] = theta[i] + dt / 10 * omega
    theta[i + 1] = np.mod(theta[i + 1], 2 * np.pi)
    
    # return level of light for each sensor
    
    sensor = np.zeros(2, )
    
    sensor[0] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    sensor[1] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    return sensor, v_motor[i, :]

def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    return v

def getObservation(x_agent, v_agent, v_motor, theta, v, w, z, a, iteration):
    x = f(x_agent, v_agent, v_motor, theta, v, w, a, iteration)
    return (g(x, v), g(x, v) + z)

def getObservationFE(x_agent, v_agent, v_motor, theta, v, w, z, a, iteration):
    x, v_motor = fFE(x_agent, v_agent, v_motor, theta, v, w, a, iteration)
    return (g(x, v), g(x, v) + z, v_motor)

def sensoryErrors(y, mu_x, mu_v, mu_gamma_z):
    eps_z = y - g_gm(mu_x, mu_v)
    pi_gamma_z = np.exp(mu_gamma_z) * np.ones((obs_states, ))
    xi_z = pi_gamma_z * eps_z
    return eps_z, xi_z


def dynamicsErrors(mu_x, mu_v, mu_gamma_w):
    eps_w = mu_x - f_gm(mu_x, mu_v)
    pi_gamma_w = np.exp(mu_gamma_w) * np.ones((obs_states, ))
    xi_w = pi_gamma_w * eps_w
    return eps_w, xi_w
#
#def priorErrors(mu_v, eta):
#    eps_n = mu_v[:, :-1] - eta
#    xi_n = pi_n * eps_n
#    return eps_n, xi_n


def FreeEnergy(y, mu_x, mu_v, mu_gamma_z, mu_gamma_w, eta):
    eps_z, xi_z = sensoryErrors(y, mu_x, mu_v, mu_gamma_z)
    eps_w, xi_w = dynamicsErrors(mu_x, mu_v, mu_gamma_w)
#    eps_n, xi_n = priorErrors(mu_v, eta)
    return .5 * (np.trace(np.dot(eps_z[:, None], np.transpose(xi_z[:, None]))) +
                 np.trace(np.dot(eps_w[:, None], np.transpose(xi_w[:, None]))) +
#                 np.trace(np.dot(eps_n, np.transpose(xi_n))) -
                 np.log(np.prod(np.exp(mu_gamma_z)) *
                        np.prod(np.exp(mu_gamma_w)))) #*
#                        np.prod(pi_n)))

def BraitenbergFreeEnergy2(noise_level, sensor_confidence, prior_confidence, motor_confidence, z1, photoaxis_first):
    s = np.zeros((iterations, sensors_n))
    v = np.zeros((sensors_n))
    x_light = np.array([59.,47.])
    theta = np.zeros((iterations, ))                            # orientation of the agent
    x_agent = np.zeros((iterations, 2))                         # 2D world, 2 coordinates por agent position
    v_agent = np.zeros((iterations, ))
    v_motor = np.zeros((iterations, motors_n))
    
    
    ### Free Energy definition
    FE = np.zeros((iterations,))
    rho = np.zeros((iterations, obs_states))
    mu_x = np.zeros((iterations, hidden_states))
    mu_m = np.zeros((iterations, hidden_states))
    mu_v = np.zeros((iterations, hidden_causes))
    a = np.zeros((iterations, motors_n))
    
    dFdmu_x = np.zeros((hidden_states))
    dFdmu_m = np.zeros((hidden_states))
    dFda = np.zeros((iterations, motors_n))
    drhoda = np.zeros((obs_states, motors_n))
    
    k_mu_x = 1 * np.ones(hidden_states,)
    k_mu_m = 10000 * np.ones(hidden_states,)
    k_a = 5000000 * np.ones(motors_n,)
    
    # noise on sensory input
    gamma_z = sensor_confidence * np.ones((obs_states, ))    # log-precisions
    real_gamma_z = noise_level * np.ones((obs_states, ))    # log-precisions (real world)
    pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
    real_pi_z = np.exp(real_gamma_z) * np.ones((obs_states, ))
    sigma_z = 1 / (np.sqrt(real_pi_z))
    z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()
    z = z1
    
    gamma_z_m = motor_confidence * np.ones((obs_states, ))    # log-precisions
    pi_z_m = np.exp(gamma_z_m) * np.ones((obs_states, ))
    real_pi_z_m = np.exp(32) * np.ones((obs_states, ))
    sigma_z_m = 1 / (np.sqrt(real_pi_z_m))
    z_m = (np.dot(np.diag(sigma_z_m), np.random.randn(obs_states, iterations))).transpose()
    
    # noise on motion of hidden states
    gamma_w = - 12 * np.ones((hidden_states, ))    # log-precision
    pi_w = np.exp(gamma_w) * np.ones((hidden_states, ))
    sigma_w = 1 / (np.sqrt(pi_w))
    w = (np.dot(np.diag(sigma_w), np.random.randn(obs_states, iterations))).transpose()
    
    gamma_w_m = prior_confidence * np.ones((hidden_states, ))    # log-precision
    pi_w_m = np.exp(gamma_w_m) * np.ones((hidden_states, ))
    sigma_w_m = 1 / (np.sqrt(pi_w_m))
    w_m = (np.dot(np.diag(sigma_w_m), np.random.randn(obs_states, iterations))).transpose()



    ### initialisation
    v = np.array([l_max, l_max])
    mu_v[0, :] = v
#    a[0, :] = np.array([30, 20])
#    mu_x[0, :] = v
    #
    ##drhoda = np.array([[0., 1.], [1., 0.]])
#    drhoda = - np.array([[1., 0.], [0., 1.]])             # vehicle 3a - lover
    drhoda = np.array([[0., 1.], [1., 0.]])             # vehicle 2b - aggressor
    x_agent[0, :] = np.array([10., 10.])
    #
    ##theta[0] = np.pi * np.random.rand()
    theta[0] = np.pi / 2
    
    # online plot routine
#    fig = plt.figure(0)
#    plt.ion()
#        
##    plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#    
#    orientation_endpoint = x_agent[0, :, None] + length_dir * (np.array([[np.cos(theta[0])], [np.sin(theta[0])]]))
#    orientation = np.concatenate((x_agent[0, :, None], orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation
#    
#    plt.xlim((0,100))
#    plt.ylim((0,200))
#    
#    # update the plot through objects
#    ax = fig.add_subplot(111)
#    line1, = ax.plot(x_agent[0, 0], x_agent[0, 1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
#    line2, = ax.plot(orientation[0, :], orientation[1, :], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma
    
    
    for i in range(iterations - 1):
        if photoaxis_first:
            if i > iterations / 4:
                gamma_z = sensor_confidence
                pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
                
                gamma_w_m = prior_confidence
                pi_w_m = np.exp(gamma_w_m) * np.ones((hidden_states, ))
                
                gamma_z_m = motor_confidence
                pi_z_m = np.exp(gamma_z_m) * np.ones((obs_states, ))
            else:
                gamma_z = noise_level
                pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
                
                gamma_w_m = - 6
                pi_w_m = np.exp(gamma_w_m) * np.ones((hidden_states, ))
                
                gamma_z_m =  - 10
                pi_z_m = np.exp(gamma_z_m) * np.ones((obs_states, ))
            
        s[i, :], rho[i, :], v_motor[i, :] = getObservationFE(x_agent, v_agent, v_motor, theta, v, z_m[i, :], z[i, :], a[i, :], i)
        
        # update plot
#        orientation_endpoint = x_agent[i, :, None] + length_dir * (np.array([[np.cos(theta[i])], [np.sin(theta[i])]]))
#        orientation = np.concatenate((x_agent[i, :, None], orientation_endpoint), axis=1)
#        line1.set_xdata(x_agent[i, 0])
#        line1.set_ydata(x_agent[i, 1])
#        line2.set_xdata(orientation[0,:])
#        line2.set_ydata(orientation[1,:])
#        fig.canvas.draw()
#        plt.pause(0.05)
        
        FE[i] = FreeEnergy(rho[i, :], mu_x[i, :], mu_v[i, :], gamma_z, gamma_w, mu_v[i, :])
        
        # find derivatives
        dFdmu_x = pi_z * (mu_x[i, :] - s[i, :]) + pi_w * (mu_x[i, :] - mu_v[i, :])  - pi_z * z[i, :] / np.sqrt(dt)
        dFdmu_m = pi_z_m * (mu_m[i, :] - v_motor[i, :]) +  pi_w_m * (mu_m[i, :] - mu_x[i, ::-1]) - pi_z_m * z_m[i, :] / np.sqrt(dt)                 # vehicle 2b - aggressor
#        dFdmu_m = pi_z_m * (mu_m[i, :] - v_motor[i,:]) +  pi_w_m * (mu_m[i, :] - l_max + mu_x[i, :])# - pi_z_m * z_m[i, :] / np.sqrt(dt)             # vehicle 3a - lover
        dFda[i, :] = np.dot((pi_z_m * (v_motor[i, :] - mu_m[i, :]) + pi_z_m * z_m[i, :] / np.sqrt(dt)), drhoda)
        
        # update equations
        mu_x[i + 1, :] = mu_x[i, :] + dt * (- k_mu_x * dFdmu_x)
        mu_m[i + 1, :] = mu_m[i, :] + dt * (- k_mu_m * dFdmu_m)
        mu_v[i + 1, :] = mu_v[i, :]
        a[i + 1, :] = a[i, :] + dt * (- k_a * dFda[i, :])
#        a[i + 1, :] = l_max - mu_m[i, :]                                        # vehicle 3a - lover
        a[i + 1, :] = mu_m[i, ::-1]                                        # vehicle 2b - aggressor

#    plt.figure()
#    plt.plot(mu_x[:, 0])
#    f = scipy.fftpack.fft(mu_x[iterations//2:, 0])
#    plt.figure()
#    plt.semilogy(np.arange(5 * dt, T / 2, dt),  2.0 / iterations * np.abs(f[:iterations//2])[5:])

#    aa = acf(x_agent[:, 0])
#    bb = acf(x_agent[:, 1])
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(aa)
#    plt.subplot(1, 2, 2)
#    plt.plot(bb)
#    
#    print(aa[1])
#    print(bb[1])

    
#    aa = print(acf(x_agent[:, 0]))
#    print(acf(x_agent[:, 1]))
    
#    plt.figure()
#    plt.plot(x_agent[:, 0], x_agent[:, 1])
#    plt.xlim((0,80))
#    plt.ylim((0,80))
#    plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#    plt.plot(x_agent[0, 0], x_agent[0, 1], color='red', marker='o', markersize=8)
#    plt.title('Trajectory (Active Inference)')
#    
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(x_agent[:-1, 0], 'b', np.ones(iterations - 1) * x_light[0], 'r')
#    plt.subplot(1, 2, 2)
#    plt.plot(x_agent[:-1, 1], 'b', np.ones(iterations - 1) * x_light[1], 'r')
#    plt.title('Position')
        
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(rho[:-1, 0], 'b', label='Noisy signal')
#    plt.plot(s[:-1, 0], 'g', label='Signal')
#    plt.plot(mu_x[:-1, 0], 'r', label='Brain state')
##    plt.plot(mu_m[:-1, 1], 'k', label='Brain state motor')
#    plt.legend()
#    plt.subplot(1, 2, 2)
#    plt.plot(rho[:-1, 1], 'b', label='Noisy signal')
#    plt.plot(s[:-1, 1], 'g', label='Signal')
#    plt.plot(mu_x[:-1, 1], 'r', label='Brain state')
##    plt.plot(mu_m[:-1, 0], 'k', label='Brain state motor')
#    plt.legend()
#    plt.title('Light')
#    
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(mu_m[:-1, 1], 'b', label='Brain state motor')
#    plt.plot(mu_x[:-1, 0], 'r', label='Brain state sensor')
#    plt.legend()
#    plt.subplot(1, 2, 2)
#    plt.plot(mu_m[:-1, 0], 'b', label='Brain state motor')
#    plt.plot(mu_x[:-1, 1], 'r', label='Brain state sensor')
#    plt.legend()
#    plt.title('Motors')
#    
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(v_motor[:-1, 0], 'b', label='Motor1')
#    plt.plot(mu_m[:-1, 0], 'r', label='Brain state motor')
#    plt.legend()
#    plt.subplot(1, 2, 2)
#    plt.plot(v_motor[:-1, 1], 'b', label='Motor2')
#    plt.plot(mu_m[:-1, 1], 'r', label='Brain state motor')
#    plt.legend()
#    plt.title('Motors')
    
#    plt.figure()
#    plt.plot(a[:-1, 0], 'b', label='Motor1')
#    plt.plot(a[:-1, 1], 'r', label='Motor2')
#    plt.title('Actions')
#    plt.legend()
#
#    plt.figure()
#    plt.plot(dFda[:-1, 0], 'b', label='Motor1')
#    plt.plot(dFda[:-1, 1], 'r', label='Motor2')
#    plt.title('Actions\'s rate of change')
#    plt.legend()
#    
#    plt.figure()
#    plt.plot(v_motor[:-1, 0], 'b', label='Motor1')
#    plt.plot(v_motor[:-1, 1], 'r', label='Motor2')
#    plt.title('Velocity')
#    plt.legend()
    
#    plt.figure()
#    plt.plot(theta)
#    plt.title('Orientation')
#    
#    plt.figure()
#    plt.plot(FE[:int(iterations/100)])
#    plt.title('Free Energy')
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
#    light_fig = plt.figure()
#    light_map = plt.imshow(light, extent=(0., points, 0., points),
#               interpolation='nearest', cmap='jet')
#    cbar = light_fig.colorbar(light_map, shrink=0.5, aspect=5)
#    
#    
    return x_agent, rho, v_motor, mu_x, mu_m, FE


def BraitenbergFreeEnergy(noise_level, desired_confidence):
    s = np.zeros((iterations, sensors_n))
    v = np.zeros((sensors_n))
    x_light = np.array([59.,47.])
    theta = np.zeros((iterations, ))                            # orientation of the agent
    x_agent = np.zeros((iterations, 2))                         # 2D world, 2 coordinates por agent position
    v_agent = np.zeros((iterations, ))
    v_motor = np.zeros((iterations, motors_n))
    
    
    ### Free Energy definition
    FE = np.zeros((iterations,))
    rho = np.zeros((iterations, obs_states))
    mu_x = np.zeros((iterations, hidden_states))
    mu_v = np.zeros((iterations, hidden_causes))
    a = np.zeros((iterations, motors_n))
    
    dFdmu_x = np.zeros((hidden_states))
    dFda = np.zeros((iterations, motors_n))
    drhoda = np.zeros((obs_states, motors_n))
    
    k_mu_x = .0001 * np.ones(hidden_states,)
    k_a = .001 * np.ones(motors_n,)
    
    # noise on sensory input
    gamma_z = noise_level * np.ones((obs_states, ))    # log-precisions
    pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
    sigma_z = 1 / (np.sqrt(pi_z))
    z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()
    
    # noise on motion of hidden states
    gamma_w = desired_confidence * np.ones((hidden_states, ))    # log-precision
    pi_w = np.exp(gamma_w) * np.ones((hidden_states, ))
    sigma_w = 1 / (np.sqrt(pi_w))
    w = (np.dot(np.diag(sigma_w), np.random.randn(obs_states, iterations))).transpose()



    ### initialisation
    v = np.array([l_max, l_max])
    mu_v[0, :] = v
    mu_x[0, :] = v
    #
    ##drhoda = np.array([[0., 1.], [1., 0.]])
    drhoda = np.array([[1., 0.], [0., 1.]])
    drhoda = np.array([[0., 1.], [1., 0.]])
    x_agent[0, :] = np.array([10., 10.])
    #
    ##theta[0] = np.pi * np.random.rand()
    theta[0] = np.pi / 2
    
    # online plot routine
#    fig = plt.figure(0)
#    plt.ion()
#        
##    plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#    
#    orientation_endpoint = x_agent[0, :, None] + length_dir * (np.array([[np.cos(theta[0])], [np.sin(theta[0])]]))
#    orientation = np.concatenate((x_agent[0, :, None], orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation
#    
#    plt.xlim((0,100))
#    plt.ylim((0,200))
#    
#    # update the plot through objects
#    ax = fig.add_subplot(111)
#    line1, = ax.plot(x_agent[0, 0], x_agent[0, 1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
#    line2, = ax.plot(orientation[0, :], orientation[1, :], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma
    
    
    for i in range(iterations - 1):
#        if x_agent[i - 1, 0] > 80.:
#            break
#        else:
        s[i, :], rho[i, :], v_motor[i, :] = getObservationFE(x_agent, v_agent, v_motor, theta, v, w[i, :], z[i, :], - dFda[i - 1, :] / pi_z, i)
        
        # update plot
#        orientation_endpoint = x_agent[i, :, None] + length_dir * (np.array([[np.cos(theta[i])], [np.sin(theta[i])]]))
#        orientation = np.concatenate((x_agent[i, :, None], orientation_endpoint), axis=1)
#        line1.set_xdata(x_agent[i, 0])
#        line1.set_ydata(x_agent[i, 1])
#        line2.set_xdata(orientation[0,:])
#        line2.set_ydata(orientation[1,:])
#        fig.canvas.draw()
#        plt.pause(0.05)
        
#        FE[i] = FreeEnergy(rho[i, :], mu_x[i, :], mu_v[i, :], gamma_z, gamma_w, mu_v[i, :])
        
        # find derivatives
        dFdmu_x = pi_z * (mu_x[i, :] - s[i, :]) + pi_w * (mu_x[i, :] - mu_v[i, :]) - pi_z * z[i, :] / np.sqrt(dt)
        dFda[i, :] = np.dot((pi_z * (s[i, :] - mu_x[i, :]) + pi_z * z[i, :] / np.sqrt(dt)), drhoda)
    #    dFda = (pi_z * (s[i, :] - mu_x[i, :]) + pi_z * z[i, :] / np.sqrt(dt)) * (sigmoid(a[i, ::-1]) * (1 - sigmoid(a[i, ::-1])))       # cyclic behaviour?
    #    dFda = (pi_z * (s[i, :] - mu_x[i, :]) + pi_z * z[i, :] / np.sqrt(dt)) * - (1 - sigmoid(a[i, :] / l_max) ** 2) / l_max
        
        # update equations
        mu_x[i + 1, :] = mu_x[i, :] + dt * (- k_mu_x * dFdmu_x)
        mu_v[i + 1, :] = mu_v[i, :]
        a[i + 1, :] = a[i, :] + dt * (- k_a * dFda[i, :])

    plt.figure()
    plt.plot(x_agent[:, 0], x_agent[:, 1])
#    plt.xlim((0,100))
#    plt.ylim((0,100))
    plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
    plt.plot(x_agent[0, 0], x_agent[0, 1], color='red', marker='o', markersize=8)
#    
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(x_agent[:-1, 0], 'b', np.ones(iterations - 1) * x_light[0], 'r')
#    plt.subplot(1, 2, 2)
#    plt.plot(x_agent[:-1, 1], 'b', np.ones(iterations - 1) * x_light[1], 'r')
#    plt.title('Position')
#        
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(rho[:-1, 0], 'b', label='Noisy signal')
    plt.plot(s[:-1, 0], 'g', label='Signal')
    plt.plot(mu_x[:-1, 0], 'r', label='Brain state')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(rho[:-1, 1], 'b', label='Noisy signal')
    plt.plot(s[:-1, 1], 'g', label='Signal')
    plt.plot(mu_x[:-1, 1], 'r', label='Brain state')
    plt.legend()
    
    plt.figure()
    plt.plot(a[:-1, 0], 'b', label='Motor1')
    plt.plot(a[:-1, 1], 'r', label='Motor2')
    plt.title('Actions')
    plt.legend()
#    
#    dFda /= pi_z
    plt.figure()
    plt.plot(dFda[:-1, 0], 'b', label='Motor1')
    plt.plot(dFda[:-1, 1], 'r', label='Motor2')
    plt.title('Actions\'s rate of change')
    plt.legend()
    
    plt.figure()
    plt.plot(v_motor[:-1, 0], 'b', label='Motor1')
    plt.plot(v_motor[:-1, 1], 'r', label='Motor2')
    plt.title('Velocity')
    plt.legend()
    
#    plt.figure()
#    plt.plot(theta)
#    plt.title('Orientation')
#    
#    plt.figure()
#    plt.semilogy(FE)
#    plt.title('Free Energy')
#    
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
#    light_fig = plt.figure()
#    light_map = plt.imshow(light, extent=(0., points, 0., points),
#               interpolation='nearest', cmap='jet')
#    cbar = light_fig.colorbar(light_map, shrink=0.5, aspect=5)
    
    
    return x_agent

### standard Braitenberg vehicle ###

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
    
    ### initialisation
    
#    x_agent2[0, :] = np.array([10., 10. * np.random.rand()])
    x_agent2[0, :] = np.array([10., 10.])
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
    
    s2[0, :], rho2[0, :] = getObservation(x_agent2, v_agent2, v_motor2, theta2, 0., w[0, :], z[0, :], s2[0, :] + z[0, :] / np.sqrt(dt), 0)
    for i in range(1, iterations - 1):
#        if x_agent2[i - 1, 0] > 80.:
#            break
#        else:
        s2[i, :], rho2[i, :] = getObservation(x_agent2, v_agent2, v_motor2, theta2, 0., w[i, :], z[i, :], s2[i - 1, :] + z[i - 1, :] / np.sqrt(dt), i)
        
#        orientation_endpoint = x_agent2[i, :, None] + length_dir * (np.array([[np.cos(theta2[i])], [np.sin(theta2[i])]]))
#        orientation = np.concatenate((x_agent2[i, :, None], orientation_endpoint), axis=1)
#        line1.set_xdata(x_agent2[i, 0])
#        line1.set_ydata(x_agent2[i, 1])
#        line2.set_xdata(orientation[0,:])
#        line2.set_ydata(orientation[1,:])
#        fig.canvas.draw()
#        plt.pause(0.05)

#    aa = acf(x_agent2[:, 0])
#    bb = acf(x_agent2[:, 1])
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.plot(aa)
#    plt.subplot(1, 2, 2)
#    plt.plot(bb)
#    
#    print(aa[1])
#    print(bb[1])
    
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
    return x_agent2, rho2

def acf(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


motor_confidence = - 10.
desire_confidence = np.array([- 12., - 12., 0., 3. ])
motor_learning_rate = 5000000
motor_brain_state_learning_rate = 1000000
noise_level = 2
gamma_z = noise_level * np.ones((obs_states, ))    # log-precisions
pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
real_pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
sigma_z = 1 / (np.sqrt(real_pi_z))
z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()
#
#agent_position, rho = Braitenberg(noise_level, desire_confidence[0], z)
#agent_position2, rho2, mu_x2, mu_m2 = BraitenbergFreeEnergy2(noise_level, desire_confidence[1], motor_confidence, motor_learning_rate, motor_brain_state_learning_rate, z)
#agent_position3, rho3, mu_x3, mu_m3 = BraitenbergFreeEnergy2(noise_level, desire_confidence[2], motor_confidence, motor_learning_rate, motor_brain_state_learning_rate, z)
#agent_position4, rho4, mu_x4, mu_m4 = BraitenbergFreeEnergy2(noise_level, desire_confidence[3], motor_confidence, motor_learning_rate, motor_brain_state_learning_rate, z)
##
##
#### plots for paper ###
#x_light = np.array([59.,47.])
#
#plt.figure()
#plt.subplot(2, 2, 1)
#plt.plot(agent_position[:, 0], agent_position[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Braitenberg)')
#
#plt.subplot(2, 2, 2)
#plt.plot(agent_position2[:, 0], agent_position2[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position2[0, 0], agent_position2[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(-' + str(12) + ')')
#
#plt.subplot(2, 2, 3)
#plt.plot(agent_position3[:, 0], agent_position3[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(0) + ')')
#
#plt.subplot(2, 2, 4)
#plt.plot(agent_position4[:, 0], agent_position4[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position4[0, 0], agent_position4[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(3) + ')')
#
#plt.figure()
#
#plt.subplot(2, 2, 1)
#plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Braitenberg)')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 2)
#plt.plot(np.arange(0, T-dt, dt), rho2[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(-' + str(12) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 3)
#plt.plot(np.arange(0, T-dt, dt), rho3[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(0) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(0, T-dt, dt), rho4[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), mu_x4[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(3) + ')')
#plt.legend(loc = 4)
#

#motor_confidence = np.array([- 10., - 2., 5., 12. ])
#motor_learning_rate = np.array([5000000., 5000., 100., .01])
#motor_brain_state_learning_rate = np.array([1000000., 8000., 10, .01])
#desire_confidence = - 12.
#noise_level = 2
#
#agent_position, rho, mu_x, mu_m = BraitenbergFreeEnergy2(noise_level, desire_confidence, motor_confidence[0], motor_learning_rate[0], motor_brain_state_learning_rate[0], z)
#agent_position2, rho2, mu_x2, mu_m2 = BraitenbergFreeEnergy2(noise_level, desire_confidence, motor_confidence[1], motor_learning_rate[1], motor_brain_state_learning_rate[1], z)
#agent_position3, rho3, mu_x3, mu_m3 = BraitenbergFreeEnergy2(noise_level, desire_confidence, motor_confidence[2], motor_learning_rate[2], motor_brain_state_learning_rate[2], z)
#agent_position4, rho4, mu_x4, mu_m4 = BraitenbergFreeEnergy2(noise_level, desire_confidence, motor_confidence[3], motor_learning_rate[3], motor_brain_state_learning_rate[3], z)
#
#x_light = np.array([59.,47.])
#
#plt.figure()
#plt.subplot(2, 2, 1)
#plt.plot(agent_position[:, 0], agent_position[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[0]) + ')')
#
#plt.subplot(2, 2, 2)
#plt.plot(agent_position2[:, 0], agent_position2[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position2[0, 0], agent_position2[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[1]) + ')')
#
#plt.subplot(2, 2, 3)
#plt.plot(agent_position3[:, 0], agent_position3[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[2]) + ')')
#
#plt.subplot(2, 2, 4)
#plt.plot(agent_position4[:, 0], agent_position4[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position4[0, 0], agent_position4[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[3]) + ')')
#
#
#plt.figure()
#
#plt.subplot(2, 2, 1)
#plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 0], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.title('Brain states (beliefs), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[0]) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 2)
#plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m2[:-1, 0], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.title('Brain states (beliefs), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[1]) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 3)
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m3[:-1, 0], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.title('Brain states (beliefs), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[2]) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(0, T-dt, dt), mu_x4[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m4[:-1, 0], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.title('Brain states (beliefs), $\pi_{w_3} = \pi_{w_4} = $ exp(' + str(motor_confidence[3]) + ')')
#plt.legend(loc = 4)
#plt.suptitle('Left sensor')


#motor_confidence = - 10.
#desire_confidence = np.array([- 12., - 12., 0., 3. ])
#motor_learning_rate = 5000000
#motor_brain_state_learning_rate = 1000000
#noise_level = - 2
#gamma_z = noise_level * np.ones((obs_states, ))    # log-precisions
#pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
#real_pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
#sigma_z = 1 / (np.sqrt(real_pi_z))
#z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()
#
#agent_position, rho = Braitenberg(noise_level, desire_confidence[0], z)
##agent_position2, rho2, mu_x2, mu_m2 = BraitenbergFreeEnergy2(-2, desire_confidence[1], motor_confidence, motor_learning_rate, motor_brain_state_learning_rate, z)
##agent_position3, rho3, mu_x3, mu_m3 = BraitenbergFreeEnergy2(-3, desire_confidence[1], motor_confidence, motor_learning_rate, motor_brain_state_learning_rate, z)
##agent_position4, rho4, mu_x4, mu_m4 = BraitenbergFreeEnergy2(-4, desire_confidence[1], motor_confidence, motor_learning_rate, motor_brain_state_learning_rate, z)
##
##
#### plots for paper ###
#x_light = np.array([59.,47.])
#
#plt.figure()
#plt.subplot(2, 2, 1)
#plt.plot(agent_position[:, 0], agent_position[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Braitenberg)')
#
#plt.subplot(2, 2, 2)
#plt.plot(agent_position2[:, 0], agent_position2[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position2[0, 0], agent_position2[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(-' + str(12) + ')')
#
#plt.subplot(2, 2, 3)
#plt.plot(agent_position3[:, 0], agent_position3[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(0) + ')')
#
#plt.subplot(2, 2, 4)
#plt.plot(agent_position4[:, 0], agent_position4[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position4[0, 0], agent_position4[0, 1], color='red', marker='o', markersize=8)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Trajectory (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(3) + ')')
#
#plt.figure()
#
#plt.subplot(2, 2, 1)
#plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Braitenberg)')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 2)
#plt.plot(np.arange(0, T-dt, dt), rho2[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(-' + str(12) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 3)
#plt.plot(np.arange(0, T-dt, dt), rho3[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(0) + ')')
#plt.legend(loc = 4)
#plt.subplot(2, 2, 4)
#plt.plot(np.arange(0, T-dt, dt), rho4[:-1, 0], 'b', label='Noisy signal')
#plt.plot(np.arange(0, T-dt, dt), mu_x4[:-1, 0], 'r', label='Brain state')
#plt.xlabel('Time')
#plt.title('Luminance (Active Inference), $\pi_{w_1} = \pi_{w_2} = $ exp(' + str(3) + ')')
#plt.legend(loc = 4)





noise_level = 2.
sensor_confidence = np.array([- 12., noise_level])
prior_confidence = np.array([- 32., - 6.])
motor_confidence = np.array([- 10., - 4])


photoaxis_first = False 

agent_position, rho, rho_m, mu_x, mu_m, F = BraitenbergFreeEnergy2(noise_level, sensor_confidence[1], prior_confidence[1], motor_confidence[0], z, photoaxis_first)
agent_position2, rho2, rho_m2, mu_x2, mu_m2, foo = BraitenbergFreeEnergy2(noise_level, sensor_confidence[0], prior_confidence[1], motor_confidence[0], z, photoaxis_first)
agent_position3, rho3, rho_m3, mu_x3, mu_m3, F3 = BraitenbergFreeEnergy2(noise_level, sensor_confidence[1], prior_confidence[0], motor_confidence[1], z, photoaxis_first)
#
#
#photoaxis_first = True 
#
#agent_position4, rho4, rho_m4, mu_x4, mu_m4 = BraitenbergFreeEnergy2(noise_level, sensor_confidence[0], prior_confidence[1], motor_confidence[0], z, photoaxis_first)
#agent_position5, rho5, rho_m5, mu_x5, mu_m5 = BraitenbergFreeEnergy2(noise_level, sensor_confidence[1], prior_confidence[0], motor_confidence[1], z, photoaxis_first)



x_light = np.array([59.,47.])


F_interval = 2
plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, F_interval, dt), F[:int(F_interval / dt)])
plt.title('Free Energy')
plt.xlabel('Time (s)')

plt.figure(figsize=(5, 4))
plt.plot(agent_position[:, 0], agent_position[:, 1])
plt.xlim((0,80))
plt.ylim((0,80))
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
plt.title('Trajectory', fontsize=14)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance, Motor velocity')
plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt, dt), rho_m[:-1, 1], 'b', label='Motor reading $ρ_{m_2}$')
plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Proprioceptor $ρ_{m_2}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)




plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt, dt), rho2[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)




plt.figure(figsize=(5, 4))
plt.plot(agent_position3[:, 0], agent_position3[:, 1])
plt.xlim((0,80))
plt.ylim((0,80))
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=8)
plt.title('Trajectory', fontsize=14)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt, dt), rho3[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
plt.plot(np.arange(0, T-dt, dt), mu_m3[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance, Motor velocity')
plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)

F_interval = 2
plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, F_interval, dt), F3[:int(F_interval / dt)])
plt.title('Free Energy')
plt.xlabel('Time (s)')





#plt.figure(figsize=(18, 12))
#plt.subplot(3, 4, 1)
#plt.plot(agent_position[:, 0], agent_position[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
#plt.ylabel('Case 1 - Reference behaviour', fontsize=14)
#plt.title('Trajectories \n\n (1a)', fontsize=14)
#
#plt.subplot(3, 4, 2)
#plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'b', label='Sensor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'r', label='Brain state sensor')
##plt.xlabel('Time')
#plt.ylabel('Luminance')
#plt.title('Left sensor tracking \n\n (1b)', fontsize=14)
#plt.legend(loc = 4)
#
#plt.subplot(3, 4, 3)
#plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Luminance - Speed')
#plt.title('Left sensor - right motor matching \n\n (1c)', fontsize=14)
#plt.legend(loc = 4)
#
#plt.subplot(3, 4, 4)
#plt.plot(np.arange(0, T-dt, dt), rho_m[:-1, 1], 'b', label='Motor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Speed')
#plt.title('Left motor reflex arc \n\n (1d)', fontsize=14)
#plt.legend(loc = 4)
#
#plt.subplot(3, 4, 5)
#plt.plot(agent_position2[:, 0], agent_position2[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position2[0, 0], agent_position2[0, 1], color='red', marker='o', markersize=8)
#plt.ylabel('Case 2 - Low $\pi_{z[{l_1}]}$', fontsize=14)
#plt.title('(2a)')
#
#plt.subplot(3, 4, 6)
#plt.plot(np.arange(0, T-dt, dt), rho2[:-1, 0], 'b', label='Sensor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], 'r', label='Brain state sensor')
##plt.xlabel('Time')
#plt.ylabel('Luminance')
#plt.legend(loc = 4)
#plt.title('(2b)')
#
#plt.subplot(3, 4, 7)
#plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m2[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Luminance - Speed')
#plt.legend(loc = 4)
#plt.title('(2c)')
#
#plt.subplot(3, 4, 8)
#plt.plot(np.arange(0, T-dt, dt), rho_m2[:-1, 1], 'b', label='Motor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_m2[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Speed')
#plt.legend(loc = 4)
#plt.title('(2d)')
#
#plt.subplot(3, 4, 9)
#plt.plot(agent_position3[:, 0], agent_position3[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=8)
#plt.ylabel('Case 3 - Low $\pi_{w[{m_1}]}$', fontsize=14)
#plt.title('(3a)')
#
#plt.subplot(3, 4, 10)
#plt.plot(np.arange(0, T-dt, dt), rho3[:-1, 0], 'b', label='Sensor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'r', label='Brain state sensor')
#plt.xlabel('Time')
#plt.ylabel('Luminance')
#plt.legend(loc = 4)
#plt.title('(3b)')
#
#plt.subplot(3, 4, 11)
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m3[:-1, 1], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.ylabel('Luminance - Speed')
#plt.legend(loc = 4)
#plt.title('(3c)')
#
#plt.subplot(3, 4, 12)
#plt.plot(np.arange(0, T-dt, dt), rho_m3[:-1, 1], 'b', label='Motor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_m3[:-1, 1], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.ylabel('Speed')
#plt.legend(loc = 4)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.title('(3d)')
#
#
#
#
#
#
#plt.figure(figsize=(18, 12))
#plt.subplot(3, 4, 1)
#plt.plot(agent_position[:, 0], agent_position[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=8)
#plt.ylabel('Case 1 - Reference behaviour', fontsize=14)
#plt.title('Trajectories \n\n (1a)', fontsize=14)
#
#plt.subplot(3, 4, 2)
#plt.plot(np.arange(0, T-dt, dt), rho[:-1, 0], 'b', label='Sensor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'r', label='Brain state sensor')
##plt.xlabel('Time')
#plt.ylabel('Luminance')
#plt.title('Left sensor tracking \n\n (1b)', fontsize=14)
#plt.legend(loc = 4)
#
#plt.subplot(3, 4, 3)
#plt.plot(np.arange(0, T-dt, dt), mu_x[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Luminance - Speed')
#plt.title('Left sensor - right motor matching \n\n (1c)', fontsize=14)
#plt.legend(loc = 4)
#
#plt.subplot(3, 4, 4)
#plt.plot(np.arange(0, T-dt, dt), rho_m[:-1, 1], 'b', label='Motor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_m[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Speed')
#plt.title('Left motor reflex arc \n\n (1d)', fontsize=14)
#plt.legend(loc = 4)
#
#plt.subplot(3, 4, 5)
#plt.plot(agent_position4[:, 0], agent_position4[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position4[0, 0], agent_position4[0, 1], color='red', marker='o', markersize=8)
#plt.ylabel('Case 2 - Low $\pi_{z[{l_1}]}$', fontsize=14)
#plt.title('(2a)')
#
#plt.subplot(3, 4, 6)
#plt.plot(np.arange(0, T-dt, dt), rho4[:-1, 0], 'b', label='Sensor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_x4[:-1, 0], 'r', label='Brain state sensor')
##plt.xlabel('Time')
#plt.ylabel('Luminance')
#plt.legend(loc = 4)
#plt.title('(2b)')
#
#plt.subplot(3, 4, 7)
#plt.plot(np.arange(0, T-dt, dt), mu_x4[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m4[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Luminance - Speed')
#plt.legend(loc = 4)
#plt.title('(2c)')
#
#plt.subplot(3, 4, 8)
#plt.plot(np.arange(0, T-dt, dt), rho_m4[:-1, 1], 'b', label='Motor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_m4[:-1, 1], 'r', label='Brain state motor')
##plt.xlabel('Time')
#plt.ylabel('Speed')
#plt.legend(loc = 4)
#plt.title('(2d)')
#
#plt.subplot(3, 4, 9)
#plt.plot(agent_position5[:, 0], agent_position5[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position5[0, 0], agent_position5[0, 1], color='red', marker='o', markersize=8)
#plt.ylabel('Case 3 - Low $\pi_{w[{m_1}]}$', fontsize=14)
#plt.title('(3a)')
#
#plt.subplot(3, 4, 10)
#plt.plot(np.arange(0, T-dt, dt), rho5[:-1, 0], 'b', label='Sensor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_x5[:-1, 0], 'r', label='Brain state sensor')
#plt.xlabel('Time')
#plt.ylabel('Luminance')
#plt.legend(loc = 4)
#plt.title('(3b)')
#
#plt.subplot(3, 4, 11)
#plt.plot(np.arange(0, T-dt, dt), mu_x5[:-1, 0], 'b', label='Brain state sensor')
#plt.plot(np.arange(0, T-dt, dt), mu_m5[:-1, 1], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.ylabel('Luminance - Speed')
#plt.legend(loc = 4)
#plt.title('(3c)')
#
#plt.subplot(3, 4, 12)
#plt.plot(np.arange(0, T-dt, dt), rho_m5[:-1, 1], 'b', label='Motor reading')
#plt.plot(np.arange(0, T-dt, dt), mu_m5[:-1, 1], 'r', label='Brain state motor')
#plt.xlabel('Time')
#plt.ylabel('Speed')
#plt.legend(loc = 4)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.title('(3d)')















### Run Braitenberg
#simulations = 1
#noise_levels = 1
#noise_level_min = 1.
#noise_level_max = 6.
#noise_level_range = np.arange(noise_level_min, noise_level_max, (noise_level_max - noise_level_min) / noise_levels)
#desired_confidence = 12.
#
#first_cross = np.zeros((noise_levels, simulations))
#first_cross_avg = np.zeros((noise_levels, ))
#
#agent_position = np.zeros((noise_levels, simulations, iterations, 2))
#trajectory_length = np.zeros((noise_levels, simulations))
#trajectory_length_avg = np.zeros((noise_levels, ))
#time_around_target = np.zeros((noise_levels, simulations))
#time_around_target_avg = np.zeros((noise_levels, ))
#
#interval_min = 180.
#interval_max = 180.
#
#for i in range(noise_levels):
#    for j in range(simulations):
#        print(i, j)
#        agent_position[i, j, :, :] = Braitenberg(noise_level_range[i], desired_confidence)
#        first_cross[i, j] = np.argmax(agent_position[i, j, :, 0] > 80.)        
#        trajectory_length[i, j] = sum(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) for (x1, y1), (x2, y2) in zip(agent_position[i, j, :int(first_cross[i, j]), :], agent_position[i, j, 1:int(first_cross[i, j]), :]))
##        time_around_target[i, j] = len(agent_position[i, j, agent_position[i, j, :, 0] , 0])
#    
#    first_cross[i, first_cross[i, :] == 0] = iterations
#    first_cross_avg[i] = np.average(first_cross[i, :])
#    trajectory_length[i, trajectory_length[i, :] > 200] = 200
#    trajectory_length_avg[i] = np.average(trajectory_length[i, :])

#plt.figure()
#plt.plot(noise_level_range, first_cross_avg * dt)
#plt.xlabel('Noise level (log-precision)')
#plt.title('Average time over ' + str(simulations) + ' simulations')
#
#plt.figure()
#plt.plot(noise_level_range, trajectory_length_avg)
#plt.xlabel('Noise level (log-precision)')
#plt.title('Average trajectory length over ' + str(simulations) + ' simulations')

#
#### Run (Free Energy) Braitenberg for different levels of noise
#simulations = 1
#noise_levels = 1
#noise_level_min = 1.
#noise_level_max = 6.
#noise_level_range = np.arange(noise_level_min, noise_level_max, (noise_level_max - noise_level_min) / noise_levels)
#desired_confidence = 12.
#
#first_cross = np.zeros((noise_levels, simulations))
#first_cross_avg = np.zeros((noise_levels, ))
#
#agent_position = np.zeros((noise_levels, simulations, iterations, 2))
#trajectory_length = np.zeros((noise_levels, simulations))
#trajectory_length_avg = np.zeros((noise_levels, ))
#
#for i in range(noise_levels):
#    for j in range(simulations):
#        print(i, j)
#        agent_position[i, j, :, :] = BraitenbergFreeEnergy(noise_level_range[i], desired_confidence)
#        first_cross[i, j] = np.argmax(agent_position[i, j, :, 0] > 80.)
#        trajectory_length[i, j] = sum(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) for (x1, y1), (x2, y2) in zip(agent_position[i, j, :int(first_cross[i, j]), :], agent_position[i, j, 1:int(first_cross[i, j]), :]))
#        
#    first_cross[i, first_cross[i, :] == 0] = iterations
#    first_cross_avg[i] = np.average(first_cross[i, :])
#    trajectory_length[i, trajectory_length[i, :] > 200] = 200
#    trajectory_length_avg[i] = np.average(trajectory_length[i, :])
    
#plt.figure()
#plt.plot(noise_level_range, first_cross_avg * dt)
#plt.xlabel('Noise level (log-precision)')
#plt.title('Average time over ' + str(simulations) + ' simulations')
#
#plt.figure()
#plt.plot(noise_level_range, trajectory_length_avg)
#plt.xlabel('Noise level (log-precision)')
#plt.title('Average trajectory length over ' + str(simulations) + ' simulations')


### Run (Free Energy) Braitenberg with different confidences

#simulations = 100
#confidence_levels = 20
#confidence_level_min = 8.
#confidence_level_max = 14.
#confidence_level_range = np.arange(confidence_level_min, confidence_level_max, (confidence_level_max - confidence_level_min) / confidence_levels)
#noise_level = 1.
#
#first_cross = np.zeros((confidence_levels, simulations))
#first_cross_avg = np.zeros((confidence_levels, ))
#
#agent_position = np.zeros((confidence_levels, simulations, iterations, 2))
#trajectory_length = np.zeros((confidence_levels, simulations))
#trajectory_length_avg = np.zeros((confidence_levels, ))
#
#for i in range(confidence_levels):
#    for j in range(simulations):
#        print(i, j)
#        agent_position[i, j, :, :] = BraitenbergFreeEnergy(noise_level, confidence_level_range[i])
#        aa = BraitenbergFreeEnergy(noise_level, confidence_level_range[i])
#        first_cross[i, j] = np.argmax(agent_position[i, j, :, 0] > 80.)
#        if first_cross[i, j] == 0:
#            first_cross[i, j] = iterations
#        trajectory_length[i, j] = sum(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) for (x1, y1), (x2, y2) in zip(agent_position[i, j, :int(first_cross[i, j]), :], agent_position[i, j, 1:int(first_cross[i, j]), :]))
#        
##    first_cross[i, first_cross[i, :] == 0] = iterations
#    first_cross_avg[i] = np.average(first_cross[i, :])
#    trajectory_length[i, trajectory_length[i, :] > 200] = 200
#    trajectory_length_avg[i] = np.average(trajectory_length[i, :])
#    
#plt.figure()
#plt.plot(confidence_level_range, first_cross_avg * dt)
#plt.xlabel('Confidence level on prior (log-precision)')
#plt.title('Average time over ' + str(simulations) + ' simulations')
#
#plt.figure()
#plt.plot(confidence_level_range, trajectory_length_avg)
#plt.xlabel('Noise level (log-precision)')
#plt.title('Average trajectory length over ' + str(simulations) + ' simulations')


### Run (Free Energy) Braitenberg for different levels of noise and different confidences

#simulations = 100
#noise_levels = 10
#confidence_levels = 10
#
#noise_level_min = 1.
#noise_level_max = 6.
#noise_level_range = np.arange(noise_level_min, noise_level_max, (noise_level_max - noise_level_min) / noise_levels)
#
#confidence_level_min = 8.
#confidence_level_max = 14.
#confidence_level_range = np.arange(confidence_level_min, confidence_level_max, (confidence_level_max - confidence_level_min) / confidence_levels)
#
#first_cross = np.zeros((noise_levels, confidence_levels, simulations))
#first_cross_avg = np.zeros((noise_levels, confidence_levels))
#
#agent_position = np.zeros((noise_levels, confidence_levels, simulations, iterations, 2))
#trajectory_length = np.zeros((noise_levels, confidence_levels, simulations))
#trajectory_length_avg = np.zeros((noise_levels, confidence_levels))
#
#for i in range(noise_levels):
#    for j in range(confidence_levels):
#        for k in range(simulations):
#            print(i, j, k)
#            agent_position[i, j, k, :, :] = BraitenbergFreeEnergy(noise_level_range[i], confidence_level_range[j])
#            first_cross[i, j, k] = np.argmax(agent_position[i, j, k, :, 0] > 80.)
#            trajectory_length[i, j, k] = sum(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) for (x1, y1), (x2, y2) in zip(agent_position[i, j, k, :int(first_cross[i, j, k]), :], agent_position[i, j, k, 1:int(first_cross[i, j, k]), :]))
#        
#        first_cross[i, j, first_cross[i, j, :] == 0] = iterations
#        first_cross_avg[i, j] = np.average(first_cross[i, j, :])
##        trajectory_length[i, j, trajectory_length[i, j, :] > 200] = 200
#        trajectory_length_avg[i, j] = np.average(trajectory_length[i, j, :])
#    
#confidence_level_range, noise_level_range = np.meshgrid(confidence_level_range, noise_level_range)
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(noise_level_range, confidence_level_range, first_cross_avg * dt, rstride=1, cstride=1, cmap='jet', alpha = .9)
#ax.set_xlabel('Noise_level_range')
#ax.set_ylabel('Confidence_level_range')
#plt.title('Average time over ' + str(simulations) + ' simulations')
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(noise_level_range, confidence_level_range, trajectory_length_avg, rstride=1, cstride=1, cmap='jet', alpha = .9)
#ax.set_xlabel('Noise_level_range')
#ax.set_ylabel('Confidence_level_range')
#plt.title('Average trajectory length over ' + str(simulations) + ' simulations')










