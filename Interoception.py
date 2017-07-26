#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:32:48 2017

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

dt_brain = .05
dt_world = .005
T_brain = 60
T_world = T_brain / 10
iterations = int(T_brain/dt_brain)
plt.close('all')
#np.random.seed(42)

x_light = np.array([9.,47.])
x_battery = np.array([69.,18.])

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

def sigma(x):
#    return 1 / (1 + np.exp(-x))
    return np.tanh(x)

def light_level(x_agent):
    sigma_x = 30.
    sigma_y = 30.
    Sigma = np.array([[sigma_x ** 2, 0.], [0., sigma_y ** 2]])
    mu = x_light
    corr = Sigma[0, 1] / (sigma_x * sigma_y)
    
    return 5655 * l_max / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - corr ** 2)) * np.exp(
            - 1 / (2 * (1 - corr ** 2)) * ((x_agent[0] - mu[0]) ** 2 / 
            (sigma_x ** 2) + (x_agent[1] - mu[1]) ** 2 / (sigma_y ** 2) - 
            2 * corr * (x_agent[0] - mu[0]) * (x_agent[1] - mu[1]) / (sigma_x * sigma_y)))

def charge_level(x_agent):
    sigma_x = 30.
    sigma_y = 30.
    Sigma = np.array([[sigma_x ** 2, 0.], [0., sigma_y ** 2]])
    mu = x_battery
    corr = Sigma[0, 1] / (sigma_x * sigma_y)
    
    return 5655 * l_max / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - corr ** 2)) * np.exp(
            - 1 / (2 * (1 - corr ** 2)) * ((x_agent[0] - mu[0]) ** 2 / 
            (sigma_x ** 2) + (x_agent[1] - mu[1]) ** 2 / (sigma_y ** 2) - 
            2 * corr * (x_agent[0] - mu[0]) * (x_agent[1] - mu[1]) / (sigma_x * sigma_y)))


# free energy functions
def g(x, v):
    return x

def f(x_agent, v_agent, v_motor, theta, v, w, a, i):
    v_motor[i, 0] = a[0]
    v_motor[i, 1] = a[1]
    
    # translation
    v_agent[i] = (v_motor[i, 0] + v_motor[i, 1]) / 2
    x_agent[i + 1, :] = x_agent[i, :] + dt_world * (v_agent[i] * np.array([np.cos(theta[i]), np.sin(theta[i])]))
        
    # rotation
    omega = turning_speed * np.float((v_motor[i, 1] - v_motor[i, 0]) / (2 * radius))
    theta[i + 1] = theta[i] + dt_world * omega
    theta[i + 1] = np.mod(theta[i + 1], 2 * np.pi)
    
    # return level of light for each sensor
    
    light_sensor = np.zeros(2, )
    
    light_sensor[0] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    light_sensor[1] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    
    battery_sensor = np.zeros(2, )
    
    battery_sensor[0] = charge_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    battery_sensor[1] = charge_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    return light_sensor, battery_sensor, v_motor[i, :]

def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    return v

def getObservationFE(x_agent, v_agent, v_motor, theta, v, w, z_l, z_c, a, iteration):
    x1, x2, v_motor = f(x_agent, v_agent, v_motor, theta, v, w, a, iteration)
    return (g(x1, v), g(x1, v) + z_l, g(x2, v), g(x2, v) + z_c, v_motor)

def sensoryErrors(y, mu_l, mu_v_l, mu_gamma_z):
    eps_z = y - g_gm(mu_l, mu_v_l)
    pi_gamma_z = np.exp(mu_gamma_z) * np.ones((obs_states, ))
    xi_z = pi_gamma_z * eps_z
    return eps_z, xi_z

def dynamicsErrors(mu_l, mu_v_l, mu_gamma_w):
    eps_w = mu_l - f_gm(mu_l, mu_v_l)
    pi_gamma_w = np.exp(mu_gamma_w) * np.ones((hidden_states, ))
    xi_w = pi_gamma_w * eps_w
    return eps_w, xi_w

def FreeEnergy(y, mu_l, mu_v_l, mu_gamma_z, mu_gamma_w):
    eps_z, xi_z = sensoryErrors(y, mu_l, mu_v_l, mu_gamma_z)
    eps_w, xi_w = dynamicsErrors(mu_l, mu_v_l, mu_gamma_w)
    return .5 * (np.trace(np.dot(eps_z[:, None], np.transpose(xi_z[:, None]))) +
                 np.trace(np.dot(eps_w[:, None], np.transpose(xi_w[:, None]))) +
                 np.log(np.prod(np.exp(mu_gamma_z)) *
                        np.prod(np.exp(mu_gamma_w))))

def BraitenbergFreeEnergy(noise_level, sensor_confidence, prior_confidence, motor_confidence, z1, learning_rate):
    l = np.zeros((iterations, sensors_n))
    c = np.zeros((iterations, sensors_n))
    v = np.zeros((sensors_n))
    theta = np.zeros((iterations, ))                            # orientation of the agent
    x_agent = np.zeros((iterations, 2))                         # 2D world, 2 coordinates por agent position
    v_agent = np.zeros((iterations, ))
    v_motor = np.zeros((iterations, motors_n))
    
    
    ### Free Energy definition
    FE = np.zeros((iterations,))                                # free energy function
    rho_l = np.zeros((iterations, obs_states))                  # sensory input (light)
    rho_c = np.zeros((iterations, obs_states))                  # sensory input (charge)
    mu_l = np.random.randn(iterations, hidden_states)                # hiddens states (light)
    mu_c = np.random.randn(iterations, hidden_states)                # hiddens states (charge)
    mu_m = np.zeros((iterations, hidden_states))
    mu_v_l = np.zeros((iterations, hidden_causes))
    mu_v_c = np.zeros((iterations, hidden_causes))
    a = np.zeros((iterations, motors_n))
    eps_z = np.zeros((iterations, obs_states))
    xi_z = np.zeros((iterations, obs_states))
    eps_z_m = np.zeros((iterations, motors_n))
    xi_z_m = np.zeros((iterations, motors_n))
    eps_w = np.zeros((iterations, hidden_states))
    xi_w = np.zeros((iterations, hidden_states))
    
    dFdmu_l = np.zeros((hidden_states))
    dFdmu_c = np.zeros((hidden_states))
    dFdmu_m = np.zeros((hidden_states))
    dFda = np.zeros((iterations, motors_n))
    drho_lda = np.zeros((obs_states, motors_n))
    
    k = learning_rate
    
    # noise on sensory input
    gamma_z_l = sensor_confidence * np.ones((sensors_n, ))    # log-precisions
    pi_z_l = np.exp(gamma_z_l) * np.ones((sensors_n, ))
    
    gamma_z_l = np.zeros((iterations, sensors_n))    # log-precision
    gamma_z_l[0,:] = -3 * np.ones((sensors_n, ))
    pi_z_l = np.exp(gamma_z_l) * np.ones((iterations, sensors_n))
    
    real_gamma_z_l = noise_level * np.ones((sensors_n, ))    # log-precisions (real world)
    real_pi_z_l = np.exp(real_gamma_z_l) * np.ones((sensors_n, ))
    sigma_z = 1 / (np.sqrt(real_pi_z_l))
    z_l = (np.dot(np.diag(sigma_z), np.random.randn(sensors_n, iterations))).transpose()
#    z = z1

    gamma_z_c = 3 * np.ones((sensors_n, ))    # log-precisions
    pi_z_c = np.exp(gamma_z_c) * np.ones((sensors_n, ))
    
    gamma_z_c = np.zeros((iterations, sensors_n))    # log-precision
    gamma_z_c[0,:] = 3 * np.ones((sensors_n, ))
    pi_z_c = np.exp(gamma_z_c) * np.ones((iterations, sensors_n))
    
    real_gamma_z_c = noise_level * np.ones((sensors_n, ))    # log-precisions (real world)
    real_pi_z_c = np.exp(real_gamma_z_c) * np.ones((sensors_n, ))
    sigma_z = 1 / (np.sqrt(real_pi_z_c))
    z_c = (np.dot(np.diag(sigma_z), np.random.randn(sensors_n, iterations))).transpose()
    
    gamma_z_m = motor_confidence * np.ones((motors_n, ))    # log-precisions
    pi_z_m = np.exp(gamma_z_m) * np.ones((motors_n, ))
    real_pi_z_m = np.exp(32) * np.ones((motors_n, ))
    sigma_z_m = 1 / (np.sqrt(real_pi_z_m))
    z_m = (np.dot(np.diag(sigma_z_m), np.random.randn(motors_n, iterations))).transpose()
    
    # noise on motion of hidden states
    gamma_w_l = np.zeros((iterations, hidden_states))    # log-precision
    gamma_w_l[0,:] = - 12 * np.ones((hidden_states, ))
    pi_w_l = np.zeros((iterations, hidden_states))
#    sigma_w = 1 / (np.sqrt(pi_w_l))
#    w = (np.dot(np.diag(sigma_w), np.random.randn(sensors_n, iterations))).transpose()
    
    gamma_w_m = prior_confidence * np.ones((hidden_states, ))    # log-precision
    pi_w_m = np.exp(gamma_w_m) * np.ones((hidden_states, ))
#    sigma_w_m = 1 / (np.sqrt(pi_w_m))
#    w_m = (np.dot(np.diag(sigma_w_m), np.random.randn(motors_n, iterations))).transpose()
    
    gamma_w_c = np.zeros((iterations, hidden_states))    # log-precision
    gamma_w_c[0,:] = 12 * np.ones((hidden_states, ))
    pi_w_c = np.exp(gamma_w_c) * np.ones((iterations, hidden_states))
#    sigma_w_c = 1 / (np.sqrt(pi_w_c))
#    w_c = (np.dot(np.diag(sigma_w_c), np.random.randn(motors_n, iterations))).transpose()


    ### initialisation
    drho_lda = np.array([[1., 0.], [0., 1.]])             # vehicle 2b - aggressor
    random_angle = 2 * np.pi * np.random.rand()
    random_norm = 60 + 10 * np.random.rand() - 5
#    x_agent[0, :] = x_light + np.array([random_norm * np.cos(random_angle), random_norm * np.sin(random_angle)])
#    theta[0] = np.pi * np.random.rand()
    
    k_1 = 3
    k_2 = 1.
    k_3 = 1.0
    
    for i in range(iterations - 1):
        l[i, :], rho_l[i, :], c[i, :], rho_c[i, :], v_motor[i, :] = getObservationFE(x_agent, v_agent, v_motor, theta, v, z_m[i, :], z_l[i, :], z_c[i, :], a[i, :], i)              # environment
        
        
        # body
        gamma_w_l[i, :] = k_1 * sigma(mu_c[i, :])
#        gamma_w_l[i, :] = -3
#        gamma_w_l[i, :] = 3
        pi_w_l[i, :] = np.exp(gamma_w_l[i, :])
        
        gamma_z_c[i, :] = k_1 * sigma(mu_c[i, :])
#        gamma_z_c[i,:] = -3
#        gamma_z_c[i,:] = 3
        pi_z_c[i, :] = np.exp(gamma_z_c[i, :])
        
#        gamma_z_l[i, :] = - k_1 * sigma(mu_c[i, :])
##        gamma_z_l[i,:] = 3
#        pi_z_l[i, :] = np.exp(gamma_z_l[i, :])
        
        gamma_w_c_dot = - k_2 * gamma_w_c[i, :] + k_3 * sigma(mu_c[i, :])
        gamma_w_c[i+1, :] = gamma_w_c[i, :] + dt_brain * gamma_w_c_dot
        gamma_w_c[i, :] = 3
#        gamma_w_c[i, :] = -3
        pi_w_c[i, :] = np.exp(gamma_w_c[i, :])
        
        gamma_z_l[i, :] = gamma_w_c[i, :]
        pi_z_l[i, :] = np.exp(gamma_z_l[i, :])
        
        
        # vehicle 2a - coward
        eps_z[i, :], xi_z[i, :] = sensoryErrors(rho_l[i, :], mu_l[i, :], mu_v_l[i, :], gamma_z)
        eps_z_m[i, :], xi_z_m[i, :] = sensoryErrors(v_motor[i, :], mu_m[i, :], mu_v_l[i, :], gamma_z_m)
#        eps_w_m[i, :], xi_w[i, :] = dynamicsErrors(mu_l[i, :], mu_c[i, :], mu_m[i, :], gamma_w_m)
        
        FE[i] = .5 * (np.dot(eps_z[i, :], np.transpose(xi_z[i, :])) + np.dot(eps_w[i, :], np.transpose(xi_w[i, :])) + np.dot(eps_z_m[i, :], np.transpose(xi_z_m[i, :]))) + np.log(np.prod(np.exp(gamma_z)) * np.prod(np.exp(gamma_z_m)) * np.prod(np.exp(gamma_w_m)))
        
        # find derivatives
        dFdmu_l = pi_z_l[i, :] * (mu_l[i, :] - l[i, :]) + pi_w_l[i, :] * (mu_l[i, :] - mu_v_l[i, :]) - pi_w_m * (mu_m[i, :] - mu_l[i, ::-1] - mu_c[i, ::-1]) - pi_z_l[i, :] * z_l[i, :] / np.sqrt(dt_brain)            # vehicle 2b - aggressor
        dFdmu_c = pi_z_c[i, :] * (mu_c[i, :] - c[i, :]) + pi_w_c[i, :] * (mu_c[i, :] - mu_v_c[i, :]) - pi_w_m * (mu_m[i, :] - mu_l[i, ::-1] - mu_c[i, ::-1]) - pi_z_c[i, :] * z_c[i, :] / np.sqrt(dt_brain)
#        dFdmu_c = pi_z_c * (mu_c[i, :] - c[i, :]) + pi_w_c[i, 0] * (mu_c[i, :] - mu_v_c[i, :])
        dFdmu_c = pi_w_c[i, 0] * (mu_c[i, :] - mu_v_c[i, :])
        dFdmu_m = pi_z_m * (mu_m[i, :] - v_motor[i, :]) +  pi_w_m * (mu_m[i, :] - mu_l[i, ::-1] - mu_c[i, ::-1]) - pi_z_m * z_m[i, :] / np.sqrt(dt_brain)                                  # vehicle 2b - aggressor
#        dFda[i, :] = np.dot((pi_z_m * (v_motor[i, :] - mu_m[i, :]) + pi_z_m * z_m[i, :] / np.sqrt(dt_brain)), drho_lda)
        
        # update equations
        mu_l[i + 1, :] = mu_l[i, :] + dt_brain * (- k * dFdmu_l)
        mu_c[i + 1, :] = mu_c[i, :] + dt_brain * (- k * dFdmu_c)
        mu_m[i + 1, :] = mu_m[i, :] + dt_brain * (- k * dFdmu_m)
#        a[i + 1, :] = a[i, :] + dt_brain * (- k * dFda[i, :])
#        mu_l[i, :] = (pi_z_l * rho_l[i, :] + pi_w_m * mu_m[i, ::-1]) / (pi_z_l + pi_w_m)
#        mu_m[i + 1, :] = (pi_z_m * v_motor[i, :] + pi_w_m * mu_l[i, ::-1]) / (pi_z_m + pi_w_m)
#        mu_l[i, :] = (pi_z_l * (s[i, :] + z[i, :] / np.sqrt(dt_brain)) + pi_w_m * mu_m[i, ::-1]) / (pi_z_l + pi_w_m)
#        mu_m[i + 1, :] = (pi_z_m * (v_motor[i, :] + z_m[i, :] / np.sqrt(dt_brain)) + pi_w_m * mu_l[i, ::-1]) / (pi_z_m + pi_w_m)
        a[i + 1, :] = mu_m[i, :]                                        # vehicle 2b - aggressor
#        mu_l[i + 1, :] = (pi_z_l * rho_l[i, :] + pi_w_l * mu_l[i, ::-1]) / (pi_z_l + pi_w_l)
#        mu_l[i + 1, :] = (pi_z_l * s[i, :] + z[i, :] / np.sqrt(dt_brain) + pi_w_l * mu_l[i, ::-1]) / (pi_z_l + pi_w_l)
        
    return x_agent, l, rho_l, c, rho_c, v_motor, mu_l, mu_c, mu_m, FE, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, theta[0], gamma_w_l, gamma_w_c, gamma_z_c, gamma_z_l


noise_level = 3.
gamma_z = noise_level * np.ones((sensors_n, ))    # log-precisions
pi_z_l = np.exp(gamma_z) * np.ones((sensors_n, ))
real_pi_z_l = np.exp(gamma_z) * np.ones((sensors_n, ))
sigma_z = 1 / (np.sqrt(real_pi_z_l))
z = (np.dot(np.diag(sigma_z), np.random.randn(sensors_n, iterations))).transpose()

sensor_confidence = np.array([- 12., noise_level])
prior_confidence = np.array([- 4., noise_level - 1.])
motor_confidence = np.array([noise_level - 12, 0.])
learning_rate = 1           # photoaxis
#learning_rate = .5          # pathological

agent_position10, s, rho_l, c, rho_c, rho_m, mu_l, mu_c, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle10, gamma_w_l, gamma_w_c, gamma_z_c, gamma_z_l = BraitenbergFreeEnergy(noise_level, sensor_confidence[1], prior_confidence[1], motor_confidence[0], z, learning_rate)          # phototaxis
#perturbation = .2 * np.random.randn(1, 3)
#agent_position11, s, rho_l, rho_m, mu_l, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle11 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[1]+perturbation[0,1], motor_confidence[0]+perturbation[0,2], z, learning_rate)          # phototaxis
#perturbation = .2 * np.random.randn(1, 3)
#agent_position12, s, rho_l, rho_m, mu_l, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle12 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[1]+perturbation[0,1], motor_confidence[0]+perturbation[0,2], z, learning_rate)          # phototaxis


#F_interval = .2
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, F_interval, dt_world), F[:int(F_interval / dt_world)])
#plt.title('Free Energy')
#plt.xlabel('Time (s)')
#
plt.figure(figsize=(5, 4))
plt.plot(agent_position10[:, 0], agent_position10[:, 1], color='blue')
#plt.xlim((0,80))
#plt.ylim((0,80))
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
plt.plot(x_battery[0], x_battery[1], color='green', marker='o', markersize=20)
plt.plot(agent_position10[0, 0], agent_position10[0, 1], color='red', marker='o', markersize=15)

orientation_endpoint = agent_position10[0, :] + 4*(np.array([np.cos(initial_angle10), np.sin(initial_angle10)]))
plt.plot([agent_position10[0, 0], orientation_endpoint[0]], [agent_position10[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
plt.title('Trajectory', fontsize=14)


plt.figure(figsize=(5, 4))
plt.subplot(2,2,1)
plt.plot(np.arange(0, T_world-dt_world, dt_world), rho_l[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), s[:-1, 0], 'k', label='Sensory reading $ρ_{l_1}$, no noise')
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_l[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
#plt.xlabel('Time (s)')
plt.xticks([])
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)


plt.subplot(2,2,2)
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_l[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance, Motor velocity')
plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)


plt.subplot(2,2,3)
plt.plot(np.arange(0, T_world-dt_world, dt_world), rho_c[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), s[:-1, 0], 'k', label='Sensory reading $ρ_{l_1}$, no noise')
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_c[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
#plt.xlabel('Time (s)')
plt.xticks([])
plt.ylabel('Charge')
plt.title('Exteroceptor $ρ_{c_1}$, $\mu_{c_1}$', fontsize=14)
plt.legend(loc = 4)


plt.subplot(2,2,4)
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_c[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Charge, Motor velocity')
plt.title('Beliefs $\mu_{c_1}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)


plt.figure()
plt.plot(gamma_w_l)
plt.title('Gamma_w_l', fontsize=14)

plt.figure()
plt.plot(gamma_w_c)
plt.title('Gamma_w_c', fontsize=14)

plt.figure()
plt.plot(gamma_z_c)
plt.title('Gamma_z_c', fontsize=14)

plt.figure()
plt.plot(gamma_z_l)
plt.title('Gamma_z_l', fontsize=14)

plt.figure()
plt.plot(mu_c)
plt.title('MU_c', fontsize=14)

#points = 100
#x_map = range(points)
#y_map = range(points)
#light = np.zeros((points, points))
#
#for i in range(points):
#    for j in range(points):
#        light[i, j] = light_level(np.array([x_map[j], y_map[points-i-1]])) + sigma_z[0] * np.random.randn()
#
#light_fig = plt.figure()
#light_map = plt.imshow(light, extent=(0., points, 0., points),
#           interpolation='nearest', cmap='jet')
#cbar = light_fig.colorbar(light_map, shrink=0.5, aspect=5)
#
#points = 100
#x_map = range(points)
#y_map = range(points)
#charge = np.zeros((points, points))
#
#for i in range(points):
#    for j in range(points):
#        charge[i, j] = charge_level(np.array([x_map[j], y_map[points-i-1]])) + sigma_z[0] * np.random.randn()
#
#charge_fig = plt.figure()
#charge_map = plt.imshow(charge, extent=(0., points, 0., points),
#           interpolation='nearest', cmap='jet')
#cbar = charge_fig.colorbar(charge_map, shrink=0.5, aspect=5)





