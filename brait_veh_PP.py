# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:23:22 2016

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
sensors_n = 2
motors_n = 2
variables = sensors_n + motors_n
sensor = np.zeros((sensors_n,))

# network (brain)
temp_orders = 1
nodes = 3

x = np.zeros((iterations,nodes,temp_orders))
x_init = np.random.standard_normal(nodes,)

n = .0*np.random.standard_normal((iterations,nodes))
n = np.zeros((iterations,nodes))
w_orig = np.random.standard_normal((nodes,nodes))

# perceptual inference
rho = np.zeros((variables,temp_orders))

mu_x = np.zeros((variables,temp_orders))
mu_d = np.array([[2],[2]])

eps_z = np.zeros((variables,temp_orders))
eps_w = np.zeros((motors_n,temp_orders))
eps_w2 = np.zeros((sensors_n,temp_orders))
xi_z = np.zeros((variables,temp_orders))
xi_w = np.zeros((motors_n,temp_orders))
xi_w2 = np.zeros((sensors_n,temp_orders))
pi_z = 1000*np.ones((variables,temp_orders))
pi_z[sensors_n:variables,0] *= 100
pi_w = 1000*np.ones((motors_n,temp_orders))
pi_w2 = 1000*np.ones((sensors_n,temp_orders))
sigma_z = 1/(np.sqrt(pi_z))
sigma_w = 1/(np.sqrt(pi_w))
sigma_w2 = 1/(np.sqrt(pi_w2))

FE = np.zeros((iterations,))

# active inference
a = np.zeros((motors_n,temp_orders))

# noise
z = np.zeros((variables,iterations))
z[0,:] = sigma_z[0,0]*np.random.randn(1,iterations)
z[1,:] = sigma_z[1,0]*np.random.randn(1,iterations)

# fluctuations
w = np.zeros((motors_n,iterations))
w[0,:] = sigma_w[0,0]*np.random.randn(1,iterations)
w[1,:] = sigma_w[1,0]*np.random.randn(1,iterations)

w2 = np.zeros((sensors_n,iterations))
w2[0,:] = sigma_w2[0,0]*np.random.randn(1,iterations)
w2[1,:] = sigma_w2[1,0]*np.random.randn(1,iterations)

# data (history)
pos_centre_history = np.zeros((2,iterations))
vel_centre_history = np.zeros((1,iterations))
vel_history = np.zeros((2,iterations))
theta_history = np.zeros((1,iterations))
orientation_history = np.zeros((2,2,iterations))
sensor_history = np.zeros((sensors_n,iterations))
rho_history = np.zeros((variables,iterations))                  # noisy version of the sensors

mu_x_history = np.zeros((iterations,variables,temp_orders))
mu_d_history = np.zeros((iterations,sensors_n,temp_orders))
a_history = np.zeros((iterations,motors_n))

### environment ###

# light source
pos_centre_light = np.array([39,47])
light_intensity = 200

def light_level(point):
    distance = np.linalg.norm(pos_centre_light - point)
    return light_intensity/(distance**2)
    
def f(sensed_value):
    # vehicles 2
    return np.tanh(sensed_value)

    # vehicles 3
#    return .5*(1-np.tanh(sensed_value))

def dfdmu_x(sensed_value):
    # vehicles 2
    return (1 - np.tanh(sensed_value)**2)

    # vehicles 3
#    return .5*(np.tanh(sensed_value)**2)


### plot ###

# plot initial position
plt.close('all')
fig = plt.figure(0)
    
plt.plot(pos_centre_light[0], pos_centre_light[1], color='orange', marker='o', markersize=20)

orientation_endpoint = pos_centre + length_dir*(np.array([[np.cos(theta)], [np.sin(theta)]]))
orientation = np.concatenate((pos_centre,orientation_endpoint), axis=1)                            # vector containing centre of mass and endpoint for the line representing the orientation

plt.xlim((0,100))
plt.ylim((0,100))

# update the plot thrpugh objects
ax = fig.add_subplot(111)
line1, = ax.plot(pos_centre[0], pos_centre[1], color='lightblue', marker='.', markersize=30*radius)       # Returns a tuple of line objects, thus the comma
line2, = ax.plot(orientation[0,:], orientation[1,:], color='black', linewidth=2)            # Returns a tuple of line objects, thus the comma


### initialise variables ###
pos_centre = np.array([[76.],[79.]])            # can't start too close or too far for some reason
pos_centre = 100*np.random.random((2,1))
pos_centre = np.array([[4.],[77.]])

vel = 2*np.random.random((2,1))-1

omega = 0
theta = np.pi*2*np.random.uniform()
theta = 4/3*np.pi

x[0,:,0] = x_init
w_orig = np.array([[ 1.12538509, -2.00524372, 0.64383674], [-0.61054784, 0.15221595, -0.36371622], [-0.02720039, 1.39925152, 0.84412855]])
alpha = 1*np.ones((nodes,))

eta_mu_x = .0001*np.ones((variables,temp_orders))
eta_a = .0001*np.ones((motors_n,1))

for i in range(iterations-1):
    print(i)
    # brain
#    x[i,:,1] = 1/alpha*(- x[i,:,0] + np.tanh(np.dot(w_orig,x[i,:,0]))) + n[i,]
#    x[i+1,:,0] = x[i,:,0] + dt*x[i,:,1]
    
    # perception
    sensor[0] = light_level(pos_centre + radius*(np.array([[np.cos(theta+sensors_angle)], [np.sin(theta+sensors_angle)]])))            # left sensor
    sensor[1] = light_level(pos_centre + radius*(np.array([[np.cos(theta-sensors_angle)], [np.sin(theta-sensors_angle)]])))            # right sensor
    
    # action
#    vel[0] = x[i,0,1]                   # attach neuron to motor
#    vel[1] = x[i,1,1]                   # attach neuron to motor
    
    # vehicle 2
    vel[0] = f(sensor[1]) + np.tanh(a[0])                   # attach neuron to motor
    vel[1] = f(sensor[0]) + np.tanh(a[1])                   # attach neuron to motor
    
    # vehicle 3
#    vel[0] = f(sensor[0])                   # attach neuron to motor
#    vel[1] = f(sensor[1])                   # attach neuron to motor
    
    # translation
    vel_centre = (vel[0]+vel[1])/2
    pos_centre += dt*(vel_centre*np.array([[np.cos(theta)], [np.sin(theta)]]))
    
    # rotation
    omega = 20*np.float((vel[1]-vel[0])/(2*radius))
    theta += dt*omega
    
    ### inference ###
    
    # add noise and fluctuations
    rho[0:sensors_n,0] = sensor + z[0:sensors_n,i]
    rho[sensors_n:variables,0] = np.squeeze(vel) + z[sensors_n:variables,i]
    mu_x[sensors_n:variables,0] += w[:,i]/100

    eps_z[:,0] = np.squeeze(rho - mu_x)
    xi_z[:,0] = pi_z[:,0]*eps_z[:,0]
    
    #mu_x[sensors_n:variables,0] += w[:,i]
    
    #eps_w[:,0] = mu_x[sensors_n:variables,0] - f(mu_x[0:sensors_n,0])
    eps_w[0,0] = mu_x[sensors_n,0] - f(mu_x[1,0])
    eps_w[1,0] = mu_x[sensors_n+1,0] - f(mu_x[0,0])
    xi_w[:,0] = pi_w[:,0]*eps_w[:,0]
    
    eps_w2[:,0] = mu_x[0:sensors_n,0] - mu_d[:,0]
    xi_w2[:,0] = pi_w2[:,0]*eps_w2[:,0]
    
    FE[i] = .5*(np.trace(np.dot(eps_z,np.transpose(xi_z))) + np.trace(np.dot(eps_w,np.transpose(xi_w))) + np.trace(np.dot(eps_w2,np.transpose(xi_w2))))
    
    # perception
    #dFdmu_x = np.transpose(np.array([xi_z[:,0]*-1 + xi_w[:,0]*-dfdmu_x(mu_x[:,0]), xi_w[:,0]]))
    dFdmu_x = np.transpose(np.array([xi_z[:,0]*-1 + np.concatenate([xi_w[:,0]*-dfdmu_x(mu_x[0:sensors_n,0]) + xi_w2[:,0], xi_w[:,0]])]))    
    mu_x += dt* -eta_mu_x*dFdmu_x
    
    # action
    dFda = np.transpose(np.array([xi_z[sensors_n:variables,0]*(1-np.tanh(a[:,0])**2)]))
    a += dt* -eta_a*dFda
    
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
    sensor_history[:,i] = sensor[:]
    rho_history[:,i] = np.squeeze(rho[:])
    
    mu_x_history[i,:,:] = mu_x
    mu_d_history[i,:,:] = mu_d
    a_history[i,:] = a[:,0]
    
plt.figure(1)
plt.plot(pos_centre_history[0,:-1], pos_centre_history[1,:-1])
plt.xlim((0,100))
plt.ylim((0,100))
plt.plot(pos_centre_light[0], pos_centre_light[1], color='orange', marker='o', markersize=20)

plt.figure(2)
plt.subplot(1,2,1)
plt.plot(range(iterations), rho_history[0,:], 'b', range(iterations), mu_x_history[:,0,0], 'r')
plt.title("Inferred light level")
plt.subplot(1,2,2)
plt.plot(range(iterations), rho_history[1,:], 'b', range(iterations), mu_x_history[:,1,0], 'r')


plt.figure(3)
plt.subplot(1,2,1)
plt.plot(range(iterations), vel_history[0,:], 'b', range(iterations), mu_x_history[:,2,0], 'r')
plt.title("Inferred speed")
plt.subplot(1,2,2)
plt.plot(range(iterations), vel_history[1,:], 'b', range(iterations), mu_x_history[:,3,0], 'r')


plt.figure(4)
plt.subplot(1,2,1)
plt.plot(range(iterations), a_history[:,0])
plt.title("Actions")
plt.subplot(1,2,2)
plt.plot(range(iterations), a_history[:,1])


plt.figure(5)
plt.subplot(1,2,1)
plt.plot(range(iterations), mu_x_history[:,0,0], 'b', range(iterations), mu_d_history[:,0,0], 'r')
plt.title("Priors")
plt.subplot(1,2,2)
plt.plot(range(iterations), mu_x_history[:,1,0], 'b', range(iterations), mu_d_history[:,1,0], 'r')




