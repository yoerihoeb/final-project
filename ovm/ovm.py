#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 2018

Optimal velocity model

@author: Paul Petersik
"""
import ovm_fc as model
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from sklearn.neighbors import KernelDensity


# =============================================================================
# choose a setup
# =============================================================================
"""
Here you can choose different setups that are defined in ovm_init.py

B95  - as in Bando et al (1995)
MS17 - as in  Miura and Sugiyama (2017)
M14  - as in Marschler et al (2014)
O05  - as in Orosz et al (2005)

free  - customized setup 
"""

setup = "free"

# =============================================================================
#  parameter dictionary is just used when setup = "free"
# =============================================================================
"""
N    - number of cars
L    - length of the circuit
a    -  sensitivity
h    - parameter in the OV function (inflection point)
tmax - integration time
dt   - time step for the numerical integration
ovf  -  OV function
n    - box size for mesoscale variables 
box  - box in front, middle or back of a car
model - OVM, OVM_rho, OVM_Delta_x_relax2J, OVM_rho_relax2J, relax2J
r      - relaxing strength
"""

parameters = {
        "N":100,
        "L":200,
        "a":1.,
        "h":2.,
        "tmax":1000,
        "dt" : 0.1,
        "vmax":2.,
        "ovf":"tanh",
        "n":2,
        "box":"back",
        "model":"OVM", # continue with rho
        "r":0.3
        }

parameters["xpert"] = np.zeros(parameters["N"])
parameters["xpert"][0] = 0.001 # position perturbation

# =============================================================================
# Model simulation()
# =============================================================================


ovm = model.ovm(setup,parameters=parameters)

ovm.initCars(option="normal")

ovm.integrate(noise=None)


# =============================================================================
# get output
# =============================================================================


x = ovm.x
dot_x = ovm.dot_x
Delta_x = ovm.Delta_x
local_rho = ovm.local_rho
local_flow = ovm.local_flow
local_q = ovm.local_q

time = ovm.time
distance = ovm.distance

lifted_x = ovm.lifted_x

# =============================================================================
# plot
# =============================================================================


plt.close("all")

# plot the hovmöller plots
hov = True

# =============================================================================
# Panal 
# =============================================================================


fig, ax = plt.subplots(3,4)
fig.subplots_adjust(hspace=0.53, wspace=0.3)
fig.set_size_inches(12, 6)


# posistions
jump_car = 8 # just plot line of every 8th car

for j in np.arange(0,ovm.N,jump_car):
   diffx = np.roll(x[j,:],-1)-x[j,:]  
   masked_x = np.ma.array(x[j,:])
   masked_x[diffx<0] = np.ma.masked 
   ax[0,0].plot(time,masked_x,lw=0.2,c="k")
ax[0,0].set_title("car positions")
ax[0,0].set_ylabel("position")
ax[0,0].set_xlabel("time")
ax[0,0].set_ylim(0,ovm.L)
ax[0,0].set_xlim(0,ovm.tmax)


# velocities vs. car number
start = 1 # int(ovm.iters/10)
end   = int(ovm.iters-1)

ax[0,1].plot(dot_x[:,start],label="t="+str(int(ovm.time[start])))
ax[0,1].plot(dot_x[:,999],label="t="+str(int(ovm.time[1000])))
ax[0,1].set_title("velocity~car")
ax[0,1].set_xlabel("car number")
ax[0,1].set_ylabel("velocity")
ax[0,1].set_xlim(0,ovm.N)
ax[0,1].set_ylim(0,ovm.vmax)
ax[0,1].legend()

 
# phase space headway velocity
car =  0
start = 0
end   = ovm.iters
iters = end - start
jump = 3  # just plot every 3rd iteration to save time

c = np.linspace(ovm.time[start],ovm.time[end-1],iters)

ax[0,2].set_title("velocity~headway, car=" + str(car))
ax_scatter = ax[0,2].scatter(Delta_x[car,start:end:jump],dot_x[car,start:end:jump],marker="x",s=10,c=c[::jump])
ax[0,2].set_xlabel("headway")
ax[0,2].set_ylabel("velocity")
ax[0,2].set_ylim(0,ovm.vmax)
ax[0,2].set_xlim(0,5)
fig.colorbar(ax_scatter, ax=ax[0,2],label="time")


#std_deviation
ax[0,3].set_title("std($\Delta$x) ~t")
ax[0,3].plot(time,Delta_x.std(axis=0))
ax[0,3].set_xlabel("time")
ax[0,3].set_ylabel("std") 


# velocity hovmöller
if hov:
   jump = 100  # just consider every 100 iteration for the interpolation to save time
   x_data = x[:,::jump]
   dot_x_data = dot_x[:,::jump]
   t_data = time[::jump]
   lent = len(t_data)
   
   grid_x, grid_t = np.meshgrid(distance,time)
   x_point =  x_data.reshape(ovm.N*lent,1)
   t_point =  np.tile(t_data,ovm.N)
   t_point =  t_point.reshape(ovm.N*lent,1)
   points = np.concatenate((x_point,t_point),axis=1)
   dot_x_values = dot_x_data.reshape(ovm.N*lent)
   grid_dot_x = griddata(points, dot_x_values, (grid_x, grid_t), method='linear')
   
   cmap = "inferno"
   contours = np.arange(0,ovm.vmax+0.1,.1)
   cf = ax[1,0].contourf(time,distance,grid_dot_x.T,contours,cmap=cmap)
   ax[1,0].set_xlabel("time")
   ax[1,0].set_ylabel("position")
   ax[1,0].set_title("velocity")
   fig.colorbar(cf, ax=ax[1,0],label="velocity")
   
   
# density hovmöller
if hov:
   jump = 100  # just consider every 100 iteration for the interpolation to save time
   x_data = x[:,::jump]
   rho_data = local_rho[:,::jump]
   t_data = time[::jump]
   lent = len(t_data)
   
   grid_x, grid_t = np.meshgrid(distance,time)
   x_point =  x_data.reshape(ovm.N*lent,1)
   t_point =  np.tile(t_data,ovm.N)
   t_point =  t_point.reshape(ovm.N*lent,1)
   points = np.concatenate((x_point,t_point),axis=1)
   rho_values = rho_data.reshape(ovm.N*lent)
   grid_rho = griddata(points, rho_values, (grid_x, grid_t), method='linear')
   
   cmap = "viridis"
   contours = np.arange(0,3.1,.1)
   cf = ax[1,1].contourf(time,distance,grid_rho.T,contours,cmap=cmap)
   ax[1,1].set_xlabel("time")
   ax[1,1].set_ylabel("position")
   ax[1,1].set_title("local density")
   fig.colorbar(cf, ax=ax[1,1],label=r"$\rho$")


# flow velocity hovmöller
if hov:
   jump = 100  # just consider every 100 iteration for the interpolation to save time
   x_data = x[:,::jump]
   flow_data = local_flow[:,::jump]
   t_data = time[::jump]
   lent = len(t_data)
   
   grid_x, grid_t = np.meshgrid(distance,time)
   x_point =  x_data.reshape(ovm.N*lent,1)
   t_point =  np.tile(t_data,ovm.N)
   t_point =  t_point.reshape(ovm.N*lent,1)
   points = np.concatenate((x_point,t_point),axis=1)
   flow_values = flow_data.reshape(ovm.N*lent)
   grid_flow = griddata(points, flow_values, (grid_x, grid_t), method='linear')
   
   cmap = "inferno"
   contours = np.arange(0,ovm.vmax+0.1,.1)
   cf = ax[2,0].contourf(time,distance,grid_flow.T,contours,cmap=cmap)
   ax[2,0].set_xlabel("time")
   ax[2,0].set_ylabel("position")
   ax[2,0].set_title("local flow velocity")
   fig.colorbar(cf, ax=ax[2,0],label=r"flow velocity")


# flux density hovmöller
if hov:
   jump = 100  # just consider every 100 iteration for the interpolation to save time
   x_data = x[:,::jump]
   var_data = local_q[:,::jump]
   t_data = time[::jump]
   lent = len(t_data)
   
   grid_x, grid_t = np.meshgrid(distance,time)
   x_point =  x_data.reshape(ovm.N*lent,1)
   t_point =  np.tile(t_data,ovm.N)
   t_point =  t_point.reshape(ovm.N*lent,1)
   points = np.concatenate((x_point,t_point),axis=1)
   var_values = var_data.reshape(ovm.N*lent)
   grided_var = griddata(points, var_values, (grid_x, grid_t), method='linear')
   
   cmap = "rainbow"
   contours = np.arange(0.2,0.6,.02)
   cf = ax[2,1].contourf(time,distance,grided_var.T,contours,cmap=cmap,extend="both")
   ax[2,1].set_xlabel("time")
   ax[2,1].set_ylabel("position")
   ax[2,1].set_title("local flux density velocity")
   fig.colorbar(cf, ax=ax[2,1],label=r"flux density")
 
   
# fundamental diagram all
car =  0
start = 0
end   = ovm.iters
iters = end - start
jump = 3  # just plot every 3rd iteration to save time

c = dot_x[:,start:end:jump].ravel()
x = local_rho[:,start:end:jump].ravel()
y = local_q[:,start:end:jump].ravel()

rho_ideal = np.linspace(0.01,5,100)
q_ideal  = rho_ideal * ovm.V(1/rho_ideal) 

ax[2,3].set_title("fundamental diagramm")
ax_scatter = ax[2,3].scatter(x,y,marker=".",s=10,c=c,cmap="inferno")
ax[2,3].set_xlabel("density")
ax[2,3].set_ylabel("flux density")
ax[2,3].set_xlim(0,2)
ax[2,3].set_ylim(0,1.1*max(y))
fig.colorbar(ax_scatter, ax=ax[2,3],label="velocity")

ax[2,3].plot(rho_ideal,q_ideal)


# fundamental diagram one car
car =  0
start = int(0.8*ovm.iters)
end   = ovm.iters
iters = end - start
jump = 3  # just plot every 3rd iteration to save time

c = np.linspace(ovm.time[start],ovm.time[end-1],iters)
x = local_rho[car,start:end:jump]
y = local_q[car,start:end:jump]

rho_ideal = np.linspace(0.01,5,100)
q_ideal  = rho_ideal * ovm.V(1/rho_ideal) 

ax[1,3].set_title("fundamental diagramm")
ax_scatter = ax[1,3].scatter(x,y,marker="x",s=10,c=c[::jump],cmap="viridis")
ax[1,3].set_xlabel("density")
ax[1,3].set_ylabel("flux density")
ax[1,3].set_xlim(0,2)
ax[1,3].set_ylim(0,1.1*max(y))
fig.colorbar(ax_scatter, ax=ax[1,3],label="time")
ax[1,3].plot(rho_ideal,q_ideal)


#kernel densitiy 
dot_x_plot = np.linspace(0, 2, 1000)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(dot_x[:,-1][:, np.newaxis])
log_dens = kde.score_samples(dot_x_plot)

ax[1,2].plot(dot_x_plot[:, 0], np.exp(log_dens))
ax[1,2].set_title("KDE of velocities")
ax[1,2].set_ylabel("Normalized Density")
ax[1,2].set_xlabel("velocity") 
    

plt.savefig("plot.png")

del(ovm)


