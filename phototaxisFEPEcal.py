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

dt_brain = .05
dt_world = .005
T_brain = 30
T_world = T_brain / 10
iterations = int(T_brain/dt_brain)
plt.close('all')
#np.random.seed(42)


x_light = np.array([9.,37.])


F_interval = .2
plt.figure(figsize=(5, 4))
plt.plot(np.arange(0, F_interval, dt_world), F[:int(F_interval / dt_world)])
plt.title('Free Energy')
plt.xlabel('Time (s)')
#
plt.figure(figsize=(5, 4))
plt.plot(agent_position[:, 0], agent_position[:, 1], color='orange')
plt.plot(agent_position2[:, 0], agent_position2[:, 1], color='blue')
plt.plot(agent_position3[:, 0], agent_position3[:, 1], color='orange')
plt.plot(agent_position4[:, 0], agent_position4[:, 1], color='blue')
plt.plot(agent_position5[:, 0], agent_position5[:, 1], color='orange')
plt.plot(agent_position6[:, 0], agent_position6[:, 1], color='blue')
plt.plot(agent_position7[:, 0], agent_position7[:, 1], color='blue')
plt.plot(agent_position8[:, 0], agent_position8[:, 1], color='blue')
plt.plot(agent_position9[:, 0], agent_position9[:, 1], color='blue')
plt.plot(agent_position10[:, 0], agent_position10[:, 1], color='blue')
#plt.xlim((0,80))
#plt.ylim((0,80))
plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
plt.plot(agent_position[0, 0], agent_position[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position2[0, 0], agent_position2[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position3[0, 0], agent_position3[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position4[0, 0], agent_position4[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position5[0, 0], agent_position5[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position6[0, 0], agent_position6[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position7[0, 0], agent_position7[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position8[0, 0], agent_position8[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position9[0, 0], agent_position9[0, 1], color='red', marker='o', markersize=15)
plt.plot(agent_position10[0, 0], agent_position10[0, 1], color='red', marker='o', markersize=15)

orientation_endpoint = agent_position[0, :] + 4*(np.array([np.cos(initial_angle), np.sin(initial_angle)]))
plt.plot([agent_position[0, 0], orientation_endpoint[0]], [agent_position[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position2[0, :] + 4*(np.array([np.cos(initial_angle2), np.sin(initial_angle2)]))
plt.plot([agent_position2[0, 0], orientation_endpoint[0]], [agent_position2[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position3[0, :] + 4*(np.array([np.cos(initial_angle3), np.sin(initial_angle3)]))
plt.plot([agent_position3[0, 0], orientation_endpoint[0]], [agent_position3[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position4[0, :] + 4*(np.array([np.cos(initial_angle4), np.sin(initial_angle4)]))
plt.plot([agent_position4[0, 0], orientation_endpoint[0]], [agent_position4[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position5[0, :] + 4*(np.array([np.cos(initial_angle5), np.sin(initial_angle5)]))
plt.plot([agent_position5[0, 0], orientation_endpoint[0]], [agent_position5[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position6[0, :] + 4*(np.array([np.cos(initial_angle6), np.sin(initial_angle6)]))
plt.plot([agent_position6[0, 0], orientation_endpoint[0]], [agent_position6[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position7[0, :] + 4*(np.array([np.cos(initial_angle7), np.sin(initial_angle7)]))
plt.plot([agent_position7[0, 0], orientation_endpoint[0]], [agent_position7[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position8[0, :] + 4*(np.array([np.cos(initial_angle8), np.sin(initial_angle8)]))
plt.plot([agent_position8[0, 0], orientation_endpoint[0]], [agent_position8[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position9[0, :] + 4*(np.array([np.cos(initial_angle9), np.sin(initial_angle9)]))
plt.plot([agent_position9[0, 0], orientation_endpoint[0]], [agent_position9[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
orientation_endpoint = agent_position10[0, :] + 4*(np.array([np.cos(initial_angle10), np.sin(initial_angle10)]))
plt.plot([agent_position10[0, 0], orientation_endpoint[0]], [agent_position10[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
plt.title('Trajectory', fontsize=14)


plt.figure(figsize=(5, 4))
plt.subplot(2,1,1)
plt.plot(np.arange(0, T_world-dt_world, dt_world), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), s[:-1, 0], 'k', label='Sensory reading $ρ_{l_1}$, no noise')
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
#plt.xlabel('Time (s)')
plt.xticks([])
plt.ylabel('Luminance')
plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
plt.legend(loc = 4)

#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T_world-dt_world, dt_world), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), s[:-1, 0], 'k', label='Sensory reading $ρ_{l_1}$, no noise')
##plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Luminance')
#plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
#plt.legend(loc = 4)
#
plt.subplot(2,1,2)
#plt.figure(figsize=(5, 2))
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_x[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
plt.xlabel('Time (s)')
plt.ylabel('Luminance, Motor velocity')
plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
plt.legend(loc = 4)

#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, T_world-dt_world, dt_world), rho_m[:-1, 1], 'b', label='Motor reading $ρ_{m_2}$')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Velocity')
#plt.title('Proprioceptor $ρ_{m_2}$, $\mu_{m_2}$', fontsize=14)
#plt.legend(loc = 4)

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

#plt.figure()
#plt.semilogy(xi_z[:, 0], 'b', label = 'PE left light sensor')
#plt.semilogy(xi_w[:, 0], 'r', label = 'PE prior')
#plt.semilogy(xi_z_m[:, 1], 'g', label = 'PE right motor')
#plt.legend(loc = 4)

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
#perturbation_constant = .1
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position20, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle20 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position21, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle21 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position22, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle22 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position23, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle23 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position24, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle24 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position25, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle25 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position26, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle26 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position27, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle27 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#perturbation = perturbation_constant * np.random.randn(1, 3)
#agent_position28, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle28 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#agent_position29, s, rho, rho_m, mu_x, mu_m, F, eps_z, xi_z, eps_z_m, xi_z_m, eps_w, xi_w, initial_angle29 = BraitenbergFreeEnergy(noise_level, sensor_confidence[1]+perturbation[0,0], prior_confidence[0]+perturbation[0,0], motor_confidence[1]+perturbation[0,0], z, learning_rate)          # pathological
#
#F_interval = .2
#plt.figure(figsize=(5, 4))
#plt.plot(np.arange(0, F_interval, dt_world), F[:int(F_interval / dt_world)])
#plt.title('Free Energy')
#plt.xlabel('Time (s)')
#
#plt.figure(figsize=(5, 4))
#plt.plot(agent_position20[:, 0], agent_position20[:, 1], color='green')
#plt.plot(agent_position21[:, 0], agent_position21[:, 1], color='green')
#plt.plot(agent_position22[:, 0], agent_position22[:, 1], color='green')
#plt.plot(agent_position23[:, 0], agent_position23[:, 1], color='green')
#plt.plot(agent_position24[:, 0], agent_position24[:, 1], color='green')
#plt.plot(agent_position25[:, 0], agent_position25[:, 1], color='green')
#plt.plot(agent_position26[:, 0], agent_position26[:, 1], color='green')
#plt.plot(agent_position27[:, 0], agent_position27[:, 1], color='green')
#plt.plot(agent_position28[:, 0], agent_position28[:, 1], color='green')
#plt.plot(agent_position29[:, 0], agent_position29[:, 1], color='blue')
##plt.xlim((0,80))
##plt.ylim((0,80))
#plt.plot(x_light[0], x_light[1], color='orange', marker='o', markersize=20)
#plt.plot(agent_position20[0, 0], agent_position20[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position21[0, 0], agent_position21[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position22[0, 0], agent_position22[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position23[0, 0], agent_position23[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position24[0, 0], agent_position24[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position25[0, 0], agent_position25[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position26[0, 0], agent_position26[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position27[0, 0], agent_position27[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position28[0, 0], agent_position28[0, 1], color='red', marker='o', markersize=15)
#plt.plot(agent_position29[0, 0], agent_position29[0, 1], color='red', marker='o', markersize=15)
#
#orientation_endpoint = agent_position20[0, :] + 4*(np.array([np.cos(initial_angle20), np.sin(initial_angle20)]))
#plt.plot([agent_position20[0, 0], orientation_endpoint[0]], [agent_position20[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position21[0, :] + 4*(np.array([np.cos(initial_angle21), np.sin(initial_angle21)]))
#plt.plot([agent_position21[0, 0], orientation_endpoint[0]], [agent_position21[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position22[0, :] + 4*(np.array([np.cos(initial_angle22), np.sin(initial_angle22)]))
#plt.plot([agent_position22[0, 0], orientation_endpoint[0]], [agent_position22[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position23[0, :] + 4*(np.array([np.cos(initial_angle23), np.sin(initial_angle23)]))
#plt.plot([agent_position23[0, 0], orientation_endpoint[0]], [agent_position23[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position24[0, :] + 4*(np.array([np.cos(initial_angle24), np.sin(initial_angle24)]))
#plt.plot([agent_position24[0, 0], orientation_endpoint[0]], [agent_position24[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position25[0, :] + 4*(np.array([np.cos(initial_angle25), np.sin(initial_angle25)]))
#plt.plot([agent_position25[0, 0], orientation_endpoint[0]], [agent_position25[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position26[0, :] + 4*(np.array([np.cos(initial_angle26), np.sin(initial_angle26)]))
#plt.plot([agent_position26[0, 0], orientation_endpoint[0]], [agent_position26[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position27[0, :] + 4*(np.array([np.cos(initial_angle27), np.sin(initial_angle27)]))
#plt.plot([agent_position27[0, 0], orientation_endpoint[0]], [agent_position27[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position28[0, :] + 4*(np.array([np.cos(initial_angle28), np.sin(initial_angle28)]))
#plt.plot([agent_position28[0, 0], orientation_endpoint[0]], [agent_position28[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#orientation_endpoint = agent_position29[0, :] + 4*(np.array([np.cos(initial_angle29), np.sin(initial_angle29)]))
#plt.plot([agent_position29[0, 0], orientation_endpoint[0]], [agent_position29[0, 1], orientation_endpoint[1]], color='black', linewidth=2)
#plt.title('Trajectory', fontsize=14)
#
#plt.figure(figsize=(5, 4))
#plt.subplot(2,1,1)
#plt.plot(np.arange(0, T_world-dt_world, dt_world), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
##plt.plot(np.arange(0, T_world-dt_world, dt_world), s[:-1, 0], 'k', label='Sensory reading $ρ_{l_1}$, no noise')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
##plt.xlabel('Time (s)')
#plt.xticks([])
#plt.ylabel('Luminance')
#plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
#plt.legend(loc = 4)
#
##plt.figure(figsize=(5, 4))
##plt.plot(np.arange(0, T_world-dt_world, dt_world), rho[:-1, 0], 'b', label='Sensory reading $ρ_{l_1}$')
##plt.plot(np.arange(0, T_world-dt_world, dt_world), s[:-1, 0], 'k', label='Sensory reading $ρ_{l_1}$, no noise')
###plt.plot(np.arange(0, T-dt_brain, dt_brain), mu_x[:-1, 0], ':r', label='Belief about sensory reading $\mu_{l_1}$')
##plt.xlabel('Time (s)')
##plt.ylabel('Luminance')
##plt.title('Exteroceptor $ρ_{l_1}$, $\mu_{l_1}$', fontsize=14)
##plt.legend(loc = 4)
##
#plt.subplot(2,1,2)
##plt.figure(figsize=(5, 2))
#plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_x[:-1, 0], 'b', label='Belief about sensory reading $\mu_{l_1}$')
#plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
#plt.xlabel('Time (s)')
#plt.ylabel('Luminance, Motor velocity')
#plt.title('Beliefs $\mu_{l_1}$, $\mu_{m_2}$', fontsize=14)
#plt.legend(loc = 4)
#
##plt.figure(figsize=(5, 4))
##plt.plot(np.arange(0, T_world-dt_world, dt_world), rho_m[:-1, 1], 'b', label='Motor reading $ρ_{m_2}$')
##plt.plot(np.arange(0, T_world-dt_world, dt_world), mu_m[:-1, 1], ':r', label='Belief about motor reading $\mu_{m_2}$')
##plt.xlabel('Time (s)')
##plt.ylabel('Velocity')
##plt.title('Proprioceptor $ρ_{m_2}$, $\mu_{m_2}$', fontsize=14)
##plt.legend(loc = 4)