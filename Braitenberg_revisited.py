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

dt = .02
T = 4
iterations = int(T/dt)
plt.close('all')

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
dFda = np.zeros((motors_n))
drhoda = np.zeros((obs_states, motors_n))

k_mu_x = .0001 * np.ones(hidden_states,)
k_a = .1 * np.ones(motors_n,)

# noise on sensory input
gamma_z = 4 * np.ones((obs_states, ))    # log-precisions
pi_z = np.exp(gamma_z) * np.ones((obs_states, ))
sigma_z = 1 / (np.sqrt(pi_z))
z = (np.dot(np.diag(sigma_z), np.random.randn(obs_states, iterations))).transpose()

# noise on motion of hidden states
gamma_w = 12 * np.ones((hidden_states, ))    # log-precision
pi_w = np.exp(gamma_w) * np.ones((hidden_states, ))
sigma_w = 1 / (np.sqrt(pi_w))
w = (np.dot(np.diag(sigma_w), np.random.randn(obs_states, iterations))).transpose()

# functions #
def sigmoid(x):
    # vehicles 3
#    return 1 / (1 + np.exp(- 2 * x / l_max))
    return 1 * np.tanh(x / 1)

def light_level(x_agent):
#    distance = np.linalg.norm(x_light - x_agent)
#    return l_max/(distance**2)
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

def f(x_agent, v, w, a, i):    
        # vehicle 2
#    v_motor[0] =  f(a[1])
#    v_motor[1] =  f(a[0])
    
    # vehicle 3
    v_motor[i, 0] = max_speed * (sigmoid(1 - a[0]))
    v_motor[i, 1] = max_speed * (sigmoid(1 - a[1]))
    
#    v_motor[:,0] += + w
    
    # translation
    v_agent[i] = (v_motor[i, 0] + v_motor[i, 1]) / 2
    x_agent[i + 1, :] = x_agent[i, :] + dt * (v_agent[i] * np.array([np.cos(theta[i]), np.sin(theta[i])]))
    
#    print(x_agent[i, :])
    
    # rotation
    omega = 100 * np.float((v_motor[i, 1] - v_motor[i, 0]) / (2 * radius))
    theta[i + 1] = theta[i] + dt * omega
    theta[i + 1] = np.mod(theta[i + 1], 2 * np.pi)
    
    # return level of light for each sensor
    
    sensor = np.zeros(2, )
    
    sensor[0] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] + sensors_angle)], [np.sin(theta[i] + sensors_angle)]])))            # left sensor
    sensor[1] = light_level(x_agent[i, :, None] + radius * (np.array([[np.cos(theta[i] - sensors_angle)], [np.sin(theta[i] - sensors_angle)]])))            # right sensor
    
    return sensor

def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    return v

def getObservation(x_agent, v, w, z, a, iteration):
    x = f(x_agent, v, w, a, i)
    return (g(x, v), g(x, v) + z)

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

### initialisation
v = np.array([l_max, l_max])
mu_v[0, :] = v
mu_x[0, :] = v

#drhoda = np.array([[0., 1.], [1., 0.]])
drhoda = - np.array([[1., 0.], [0., 1.]])
x_agent[0, :] = np.array([29., 5.])
#x_agent[0, :] = 100 * np.random.rand(1, 2)

#theta[0] = np.pi * np.random.rand()
theta[0] = np.pi / 2 #2 / 3 * np.pi


## online plot routine
fig = plt.figure(0)
plt.ion()
    
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)

orientation_endpoint = x_agent[0, :, None] + length_dir * (np.array([[np.cos(theta[0])], [np.sin(theta[0])]]))
orientation = np.concatenate((x_agent[0, :, None], orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation

plt.xlim((0,100))
plt.ylim((0,100))

# update the plot through objects
ax = fig.add_subplot(111)
line1, = ax.plot(x_agent[0, 0], x_agent[0, 1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
line2, = ax.plot(orientation[0, :], orientation[1, :], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma


for i in range(iterations - 1):
    s[i, :], rho[i, :] = getObservation(x_agent, v, w[i, :], z[i, :], a[i, :], i)
    
    # update plot
#    if np.mod(i, 1)==0:                                                                    # don't update at each time step, too computationally expensive
    orientation_endpoint = x_agent[i, :, None] + length_dir * (np.array([[np.cos(theta[i])], [np.sin(theta[i])]]))
    orientation = np.concatenate((x_agent[i, :, None], orientation_endpoint), axis=1)
    line1.set_xdata(x_agent[i, 0])
    line1.set_ydata(x_agent[i, 1])
    line2.set_xdata(orientation[0,:])
    line2.set_ydata(orientation[1,:])
    fig.canvas.draw()
    plt.pause(0.05)
#    input("\nPress Enter to continue.")

    eps_z, xi_z = sensoryErrors(rho[i, :], mu_x[i, :], mu_v[i, :], gamma_z)
#    print(eps_z)
    
    FE[i] = FreeEnergy(rho[i, :], mu_x[i, :], mu_v[i, :], gamma_z, gamma_w, mu_v[i, :])
    
    # find derivatives
    dFdmu_x = pi_z * (mu_x[i, :] - s[i, :]) + pi_w * (mu_x[i, :] - mu_v[i, :]) - pi_z * z[i, :] / np.sqrt(dt)
    dFda = np.dot((pi_z * (s[i, :] - mu_x[i, :]) + pi_z * z[i, :] / np.sqrt(dt)), drhoda)
#    dFda = (pi_z * (s[i, :] - mu_x[i, :]) + pi_z * z[i, :] / np.sqrt(dt)) * (sigmoid(a[i, ::-1]) * (1 - sigmoid(a[i, ::-1])))       # cyclic behaviour?
#    dFda = (pi_z * (s[i, :] - mu_x[i, :]) + pi_z * z[i, :] / np.sqrt(dt)) * - (1 - sigmoid(a[i, :] / l_max) ** 2) / l_max
    
    # update equations
    mu_x[i + 1, :] = mu_x[i, :] + dt * (- k_mu_x * dFdmu_x)
    mu_v[i + 1, :] = mu_v[i, :]
    a[i + 1, :] = a[i, :] + dt * (- k_a * dFda)
    a_min = - 20
    a_max = 0
    a_norm_min = - 1
    a_norm_max = 0
    a[i + 1, :] = (a_norm_max - a_norm_min) * (a[i + 1, :] - a_min) / (a_max - a_min) + a_norm_min
    
    aa = a[i + 1, :]
    aaa = max_speed * (sigmoid(aa))
    
    

plt.figure(1)
plt.plot(x_agent[:, 0], x_agent[:, 1])
plt.xlim((0,100))
plt.ylim((0,100))
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(x_agent[0, 0], x_agent[0, 1], color='red', marker='o', markersize=8)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(x_agent[:-1, 0], 'b', np.ones(iterations - 1) * x_light[0], 'r')
plt.subplot(1, 2, 2)
plt.plot(x_agent[:-1, 1], 'b', np.ones(iterations - 1) * x_light[1], 'r')
plt.title('Position')
    
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

plt.figure()
plt.plot(v_motor[:-1, 0], 'b', label='Motor1')
plt.plot(v_motor[:-1, 1], 'r', label='Motor2')
plt.title('Velocity')
plt.legend()

plt.figure()
plt.plot(theta)
plt.title('Orientation')

plt.figure()
plt.semilogy(FE)
plt.title('Free Energy')


points = 100
x_map = range(points)
y_map = range(points)
light = np.zeros((points, points))

for i in range(points):
    for j in range(points):
        light[i, j] = light_level(np.array([x_map[i], y_map[j]])) + sigma_z[0] * np.random.randn()

light_fig = plt.figure()
light_map = plt.imshow(light.transpose(), extent=(0., points, 0., points),
           interpolation='nearest', cmap='jet')
cbar = light_fig.colorbar(light_map, shrink=0.5, aspect=5)















