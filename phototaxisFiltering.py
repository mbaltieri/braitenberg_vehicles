#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:18:22 2017

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

dt_brain = .005
dt_world = .005
T = 20
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
    
    sensor = np.zeros(2, )
    
    sensor[0] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    sensor[1] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    return sensor, v_motor[i, :]

def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    return v

def getObservationFE(x_agent, v_agent, v_motor, theta, v, w, z, a, iteration):
    x, v_motor = f(x_agent, v_agent, v_motor, theta, v, w, a, iteration)
    return (g(x, v), g(x, v) + z, v_motor)

def sensoryErrors(y, mu_x, mu_v, mu_gamma_z):
    eps_z = y - g_gm(mu_x, mu_v)
    pi_gamma_z = np.exp(mu_gamma_z) * np.ones((obs_states, ))
    xi_z = pi_gamma_z * eps_z
    return eps_z, xi_z

def dynamicsErrors(mu_x, mu_v, mu_gamma_w):
    eps_w = mu_x - f_gm(mu_x, mu_v)
    pi_gamma_w = np.exp(mu_gamma_w) * np.ones((hidden_states, ))
    xi_w = pi_gamma_w * eps_w
    return eps_w, xi_w

def FreeEnergy(y, mu_x, mu_v, mu_gamma_z, mu_gamma_w):
    eps_z, xi_z = sensoryErrors(y, mu_x, mu_v, mu_gamma_z)
    eps_w, xi_w = dynamicsErrors(mu_x, mu_v, mu_gamma_w)
    return .5 * (np.trace(np.dot(eps_z[:, None], np.transpose(xi_z[:, None]))) +
                 np.trace(np.dot(eps_w[:, None], np.transpose(xi_w[:, None]))) +
                 np.log(np.prod(np.exp(mu_gamma_z)) *
                        np.prod(np.exp(mu_gamma_w))))

def BraitenbergFreeEnergy(noise_level, sensor_confidence, prior_confidence, motor_confidence, z1, learning_rate):
    s = np.zeros((iterations, sensors_n))
    v = np.zeros((sensors_n))
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
    eps_z = np.zeros((iterations, obs_states))
    xi_z = np.zeros((iterations, obs_states))
    eps_z_m = np.zeros((iterations, motors_n))
    xi_z_m = np.zeros((iterations, motors_n))
    eps_w = np.zeros((iterations, hidden_states))
    xi_w = np.zeros((iterations, hidden_states))
    
    dFdmu_x = np.zeros((hidden_states))
    dFdmu_m = np.zeros((hidden_states))
    dFda = np.zeros((iterations, motors_n))
    drhoda = np.zeros((obs_states, motors_n))
    
    k = learning_rate
    
    # noise on sensory input
    gamma_z = sensor_confidence * np.ones((sensors_n, ))    # log-precisions
    real_gamma_z = noise_level * np.ones((sensors_n, ))    # log-precisions (real world)
    pi_z = np.exp(gamma_z) * np.ones((sensors_n, ))
    real_pi_z = np.exp(real_gamma_z) * np.ones((sensors_n, ))
    sigma_z = 1 / (np.sqrt(real_pi_z))
    z = (np.dot(np.diag(sigma_z), np.random.randn(sensors_n, iterations))).transpose()
    z = z1
    
    gamma_z_m = motor_confidence * np.ones((motors_n, ))    # log-precisions
    pi_z_m = np.exp(gamma_z_m) * np.ones((motors_n, ))
    real_pi_z_m = np.exp(32) * np.ones((motors_n, ))
    sigma_z_m = 1 / (np.sqrt(real_pi_z_m))
    z_m = (np.dot(np.diag(sigma_z_m), np.random.randn(motors_n, iterations))).transpose()
    
    # noise on motion of hidden states
    gamma_w = - 12 * np.ones((hidden_states, ))    # log-precision
    pi_w = np.exp(gamma_w) * np.ones((hidden_states, ))
    sigma_w = 1 / (np.sqrt(pi_w))
    w = (np.dot(np.diag(sigma_w), np.random.randn(sensors_n, iterations))).transpose()
    
    gamma_w_m = prior_confidence * np.ones((hidden_states, ))    # log-precision
    pi_w_m = np.exp(gamma_w_m) * np.ones((hidden_states, ))
    sigma_w_m = 1 / (np.sqrt(pi_w_m))
    w_m = (np.dot(np.diag(sigma_w_m), np.random.randn(motors_n, iterations))).transpose()


    ### initialisation
    v = np.array([l_max, l_max])
    mu_v[0, :] = v
#    a[0, :] = np.array([30, 20])
#    mu_x[0, :] = v
    #
#    drhoda = - np.array([[1., 0.], [0., 1.]])             # vehicle 3a - lover
#    drhoda = np.array([[0., 1.], [1., 0.]])             # vehicle 2b - aggressor
    drhoda = np.array([[1., 0.], [0., 1.]])             # vehicle 2b - aggressor
    x_agent[0, :] = np.array([0., 0.])
    theta[0] = np.pi / 2
    
    for i in range(iterations - 1):
        s[i, :], rho[i, :], v_motor[i, :] = getObservationFE(x_agent, v_agent, v_motor, theta, v, z_m[i, :], z[i, :], a[i, :], i)
        
        eps_z[i, :], xi_z[i, :] = sensoryErrors(rho[i, :], mu_x[i, :], mu_v[i, :], gamma_z)
        eps_z_m[i, :], xi_z_m[i, :] = sensoryErrors(v_motor[i, :], mu_m[i, :], mu_v[i, :], gamma_z_m)
        eps_w[i, :], xi_w[i, :] = dynamicsErrors(mu_x[i, :], mu_m[i, :], gamma_w_m)
        FE[i] = FreeEnergy(rho[i, :], mu_x[i, :], mu_v[i, :], gamma_z, gamma_w_m)         # no prediction errors on rho_m since velocities are implemented instantenously
        
        # find derivatives
        dFdmu_x = pi_z * (mu_x[i, :] - s[i, :] - z[i, :] / np.sqrt(dt_brain)) + pi_w * (mu_x[i, :] - mu_v[i, :]) + pi_w_m * (mu_m[i, :] - mu_x[i, ::-1])
        dFdmu_m = pi_z_m * (mu_m[i, :] - v_motor[i, :]) +  pi_w_m * (mu_m[i, :] - mu_x[i, ::-1]) - pi_z_m * z_m[i, :] / np.sqrt(dt_brain)                 # vehicle 2b - aggressor
#        dFda[i, :] = np.dot((pi_z_m * (v_motor[i, :] - mu_m[i, :]) + pi_z_m * z_m[i, :] / np.sqrt(dt_brain)), drhoda)
        
        # update equations
        mu_x[i + 1, :] = mu_x[i, :] + dt_brain * (- k * dFdmu_x)
        mu_m[i + 1, :] = mu_m[i, :] + dt_brain * (- k * dFdmu_m)
#        a[i + 1, :] = a[i, :] + dt_brain * (- k * dFda[i, :])
#        mu_x[i + 1, :] = (pi_z * rho[i, :] + pi_w_m * mu_m[i, ::-1]) / (pi_z + pi_w_m)
#        mu_m[i + 1, :] = (pi_z_m * v_motor[i, :] + pi_w_m * mu_x[i, ::-1]) / (pi_z_m + pi_w_m)
#        mu_x[i, :] = (pi_z * (s[i, :] + z[i, :] / np.sqrt(dt_brain)) + pi_w_m * mu_m[i, ::-1]) / (pi_z + pi_w_m)
#        mu_m[i + 1, :] = (pi_z_m * (v_motor[i, :] + z_m[i, :] / np.sqrt(dt_brain)) + pi_w_m * mu_x[i, ::-1]) / (pi_z_m + pi_w_m)
        a[i + 1, :] = mu_m[i, :]                                        # vehicle 2b - aggressor
#        mu_x[i + 1, :] = (pi_z * rho[i, :] + pi_w * mu_x[i, ::-1]) / (pi_z + pi_w)
#        mu_x[i + 1, :] = (pi_z * s[i, :] + z[i, :] / np.sqrt(dt_brain) + pi_w * mu_x[i, ::-1]) / (pi_z + pi_w)
        
    return x_agent, s, rho, v_motor, mu_x, mu_m, FE, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w


noise_level = - 7.
gamma_z = noise_level * np.ones((sensors_n, ))    # log-precisions
pi_z = np.exp(gamma_z) * np.ones((sensors_n, ))
real_pi_z = np.exp(gamma_z) * np.ones((sensors_n, ))
sigma_z = 1 / (np.sqrt(real_pi_z))
z = (np.dot(np.diag(sigma_z), np.random.randn(sensors_n, iterations))).transpose()

sensor_confidence = np.array([- 12., noise_level])
prior_confidence = np.array([- 32., noise_level - 0])
motor_confidence = np.array([noise_level - 0, 2.])
learning_rate = 100000

agent_position, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w = BraitenbergFreeEnergy(noise_level, sensor_confidence[1], prior_confidence[1], motor_confidence[0], z, learning_rate)
#agent_position2, rho2, rho_m2, mu_x2, mu_m2, foo = BraitenbergFreeEnergy(noise_level, sensor_confidence[0], prior_confidence[1], motor_confidence[0], z)
#agent_position3, rho3, rho_m3, mu_x3, mu_m3, F3 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1], prior_confidence[0], motor_confidence[1], z)



x_light = np.array([59.,47.])


F_interval = 2
plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, F_interval, dt_brain), F[:int(F_interval / dt_brain)])
plt.title('Free Energy')
plt.xlabel('Time (s)')

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
plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt_brain, dt_brain), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
plt.plot(np.arange(0, T-dt_brain, dt_brain), s[:-1, 0], 'g', label='Sensory reading $ρ_{l_1}$, no noise')
#plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_x[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance, Motor velocity')
plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)

plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, T-dt_brain, dt_brain), rho_m[:-1, 1], 'b', label='Motor reading $ρ_{m_2}$')
plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Proprioceptor $ρ_{m_2}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)

#points = 100
#x_map = range(points)
#y_map = range(points)
#light = np.zeros((points, points))
#
#for i in range(points):
#    for j in range(points):
#        light[i, j] = light_level(np.array([x_map[j], y_map[i]])) + sigma_z[0] * np.random.randn()
#
#light_fig = plt.figure()
#light_map = plt.imshow(light, extent=(0., points, 0., points),
#           interpolation='nearest', cmap='jet')
#cbar = light_fig.colorbar(light_map, shrink=0.5, aspect=5)

plt.figure()
plt.semilogy(xi_z[:, 0], 'b', label = 'PE left light sensor')
plt.semilogy(xi_w[:, 0], 'r', label = 'PE prior')
plt.semilogy(xi_z_m[:, 1], 'g', label = 'PE right motor')
plt.legend(loc = 4)

#
#
#
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T-dt, dt), rho2[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
#plt.plot(np.arange(0, T-dt, dt), mu_x2[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Luminance')
#plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
#plt.legend(loc = 4)
#
#
#
#
#plt.figure(figsize=(5, 4))
#plt.plot(agent_position3[:, 0], agent_position3[:, 1])
#plt.xlim((0,80))
#plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=8)
#plt.title('Trajectory', fontsize=14)
#
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T-dt, dt), rho3[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Luminance')
#plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
#plt.legend(loc = 4)
#
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T-dt, dt), mu_x3[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
#plt.plot(np.arange(0, T-dt, dt), mu_m3[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Luminance, Motor velocity')
#plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
#plt.legend(loc = 4)
#
#F_interval = 2
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, F_interval, dt), F3[:int(F_interval / dt)])
#plt.title('Free Energy')
#plt.xlabel('Time (s)')
#
#
#
#
#
#
#




