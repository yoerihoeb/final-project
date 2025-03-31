#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:43:14 2018

OVM_functions.py contains all functions for the OVM model

@author: paul
"""
import numpy as np
import ovm_init as ovm_i
# =============================================================================
# functions
# =============================================================================

class ovm(object):
    def __init__(self,setup,**kwargs):
        """
        parameters of the model simulation are setup
        setup - keyword for model setup:
            values from ovm_init.py
            · B95 - Bando et al (1995)
            
            values from kwargs['parameters']
            · free 
        """
        
        if setup!="free":
            parameters = ovm_i.parameters(setup)
        else:
            parameters = kwargs['parameters']
            
        self.N =  parameters["N"]      # number of cars
        self.L =  parameters["L"]      # length of circuit
        self.distance = np.arange(0,self.L,1) #array for distance
        
        self.a =  parameters["a"]      # sensitiviy
        self.h =  parameters["h"]      # 
        self.vmax = parameters["vmax"] # maximum velocity
        
        self.n    = parameters["n"]  # number of cars in the interaction box (must be even number, if not is n=n-1)
        self.box  = parameters["box"]
        
        self.tmax  = parameters["tmax"] # maximum time
        self.dt    = parameters["dt"]     # time step
        self.iters = abs(int(self.tmax/self.dt))
        self.time  = np.arange(0,self.tmax,self.dt)
        
        self.xpert = parameters["xpert"] # position perturbation
        
        self.ovf = parameters["ovf"] # key for the choice of the OV-function
        self.acceleration_type = parameters["model"]
        self.r = parameters["r"]
        
        # allocate functions
        self.allocate_functions()

# =============================================================================
# Routines        
# =============================================================================
    
    def allocate_functions(self):
        """
        allocate some functions (avoiding later a lot of if statements)
        """
        if self.ovf=="tanh":
            self.V=self.ovf_tanh
        if self.ovf=="hs":
            self.V=self.ovf_hs
        if self.ovf=="alg":
            self.V=self.ovf_alg
        
        if self.box =="front":
            self.density = self.density_front
            self.flow_velocity = self.flow_velocity_front
        if self.box =="middle":
            self.density = self.density_middle
            self.flow_velocity = self.flow_velocity_middle
        if self.box =="back":
            self.density = self.density_back
            self.flow_velocity = self.flow_velocity_back
            
        if self.acceleration_type == "OVM":
            self.acceleration = self.acceleration_OVM_Delta_x
        if self.acceleration_type == "OVM_rho":
            self.acceleration = self.acceleration_OVM_rho
        if self.acceleration_type == "OVM_Delta_x_relax2J":
            self.acceleration = self.acceleration_OVM_Delta_x_relax2J
        if self.acceleration_type == "relax2J":
            self.acceleration = self.acceleration_relax2J
        if self.acceleration_type == "OVM_rho_relax2J":
            self.acceleration = self.acceleration_OVM_rho_relax2J
    
    def initCars(self,**kwargs):
        """
        initialise 0th time step
        """  
        
        self.b,self.c,self.f = self.steadyStateFlow(self.L,self.N)  # free flow variable
        
        self.x       = np.zeros(shape=(self.N,self.iters)) # position
        self.dot_x   = np.zeros(shape=(self.N,self.iters)) # velocity
        self.ddot_x  = np.zeros(shape=(self.N,self.iters)) # acceleration
        self.Delta_x = np.zeros(shape=(self.N,self.iters)) # headway
        self.local_rho = np.zeros(shape=(self.N,self.iters)) # local density
        self.local_flow = np.zeros(shape=(self.N,self.iters)) # local density
        self.local_q = np.zeros(shape=(self.N,self.iters)) # local flux-density
        
        self.lifted_x = np.zeros(shape=(self.N,self.iters))
        
        if kwargs['option']=="normal":
            """
            start with prescribed positions and all cars at rest
            """
            self.x[:,0]      = np.arange(0,self.L,self.b)
            self.dot_x[:,0]  = self.c + 0.1*np.sin(2*np.pi/self.N*np.arange(self.N)) #np.random.rand(self.N)
            self.ddot_x[:,0] = 0
            
            self.x[:,0] = self.x[:,0] + self.xpert
            self.Delta_x[:,0]   = self.headway(self.x[:,0],self.L)
        
        if kwargs['option']=="lift_sigma":
            """
            Lifting operator with sigma, the standard deviation, as argument
            As in Marschler et al (2014)
            """
            p = kwargs['p']
            sigma = kwargs['sigma']
            
            #x_ref = np.arange(0,self.L,self.b) + 0.1*np.sin(2*np.pi/self.N*np.arange(self.N))
            
            Delta_x_ref = kwargs['ref']#]self.headway(x_ref,self.L)
            
            Delta_x_ref_mean = Delta_x_ref.mean()
            sigma_ref = kwargs['sigma_ref']
            
            self.Delta_x[:,0] = p*sigma/sigma_ref * (Delta_x_ref - Delta_x_ref_mean) + Delta_x_ref_mean
            
            self.x[0,0] = 0
            self.x[1:,0] = np.cumsum(self.Delta_x[:,0])[:-1]
            
            self.dot_x[:,0] = self.V(self.Delta_x[:,0])
            self.ddot_x[:,0] = 0
            
        if kwargs['option']=="lift_rho":
            """
            Lifting operator with rho, the density, as argument
            At the moment no promissing results
            """
            self.x[:,0] = self.lift_density(kwargs['rho'])
            self.Delta_x[:,0] = self.headway(self.x[:,0],self.L)
            self.dot_x[:,0] = self.V(self.Delta_x[:,0])
            self.ddot_x[:,0] = 0
        
        self.local_rho[:,0] = self.density(self.x[:,0])
        self.local_flow[:,0] = self.flow_velocity(self.dot_x[:,0])
        self.local_q[:,0] = self.local_rho[:,0] * self.local_flow[:,0]
        
        self.lifted_x[:,0] = self.lift_density(self.local_rho[:,0])
            
    def integrate(self,**kwargs):
        """
        Integrate the model (until now Semi-Euler-Scheme(?))
        
        noise:
            None - no noise
            otherwise - keyword value is the strength of the noise 
        """
        if kwargs["noise"]==None:
            for i in range(0,self.iters-1):
                self.integration_procedure(i)
        else:
            for i in range(0,self.iters-1):
                self.integration_procedure(i)
                self.Delta_x[:,i+1]  = self.Delta_x[:,i+1] \
                 + kwargs["noise"] *(self.Delta_x[:,i+1] * np.random.rand(self.N) \
                 -  np.roll(self.Delta_x[:,i+1],1) * np.random.rand(self.N) )

    def integration_procedure(self,i):
        """
        Semi-implicit Euler-Scheme for one time step
        """
        
#        self.ddot_x[:,i+1] = self.acceleration(self.Delta_x[:,i],self.dot_x[:,i])
#        
#        self.dot_x[:,i+1]  = self.dot_x[:,i] + self.ddot_x[:,i+1] * self.dt
#        self.x[:,i+1]      = self.x[:,i] + self.dot_x[:,i+1] * self.dt 
#        
#        self.x[:,i+1]      = self.x[:,i+1]%self.L
#        self.Delta_x[:,i+1]   = self.headway(self.x[:,i+1],self.L)
        
        """
        RK4
        """
        h = self.dt
        k1 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i])
        self.dot_x[:,i+1] = self.dot_x[:,i] + k1*h/2
        
        k2 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i+1])
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + k2*h/2
        k3 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i+1])
        
        self.dot_x[:,i+1] = self.dot_x[:,i] + k3*h
        k4 = self.acceleration(self.Delta_x[:,i],self.local_flow[:,i],self.local_rho[:,i],self.dot_x[:,i+1])
       
        self.dot_x[:,i+1] = self.dot_x[:,i] + h/6. * (k1 + 2*k2 + 2*k3 + k4)
        
        self.x[:,i+1]      = self.x[:,i] + self.dot_x[:,i+1] * h 
        self.x[:,i+1]      = self.x[:,i+1]%self.L
        
        # Diagnostics
        self.Delta_x[:,i+1]   = self.headway(self.x[:,i+1],self.L)
        self.local_rho[:,i+1] = self.density(self.x[:,i+1])
        self.local_flow[:,i+1] = self.flow_velocity(self.dot_x[:,i+1])
        self.local_q[:,i+1] = self.local_rho[:,i+1] * self.local_flow[:,i+1]
        
        self.lifted_x[:,i+1] = self.lift_density(self.local_rho[:,i+1])
# =============================================================================
# Functions        
# =============================================================================
#    def acceleration(self,Delta_x,local_flow,local_rho,dot_x):
#        """
#        returns the accelaration of a car
#        """
#        return self.a*(self.V(Delta_x) - dot_x) 
    
    def acceleration_OVM_Delta_x(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as function of Delta x
        """
        return self.a*(self.V(Delta_x) - dot_x) 
    
    def acceleration_OVM_rho(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as function of the local rho
        """
        return self.a*(self.V(1/local_rho) - dot_x)
    
    def acceleration_OVM_Delta_x_relax2J(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as function of Delta x and relaxed to the local flow
        """
        return self.a*(self.V(Delta_x) - dot_x) + self.r * (local_flow - dot_x)
    
    def acceleration_relax2J(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as relaxation to local flow
        """
        return self.r * (local_flow - dot_x)
    
    def acceleration_OVM_rho_relax2J(self,Delta_x,local_flow,local_rho,dot_x):
        """
        returns the accelaration of a car as relaxation to local flow
        """
        return self.a*(self.V(1/local_rho) - dot_x) + self.r * (local_flow - dot_x)
    
    def headway(self,x,L):
        Dx = np.zeros(self.N)
        Dx[:-1] = (x[1:] - x[:-1]+L)%L
        Dx[-1] = (x[0] - x[-1]+L)%L
        return Dx #(np.roll(x,-1)-x+L)%L
    
    def ovf_tanh(self,Delta_x):
        """
        OV - function as in Bando et al (1995)
        Legal velocity - V(Delta_x)
        Delta_x - headway to the car in front
        """
        return self.vmax/2.*(np.tanh(Delta_x - self.h) + np.tanh(self.h))
    
    def ovf_hs(self,Delta_x):
        """
        OV - function as in Sugiyama and Yamada (1997)
        Legal velocity - V(Delta_x)
        Delta_x - headway to the car in front
        """
        return self.vmax*(np.heaviside(Delta_x - self.h,1))
    
    def ovf_alg(self,Delta_x):
        """
        OV - function as in Orosz (2005)
        Legal velocity - V(Delta_x)
        Delta_x - headway to the car in front
        """
        ovf_return = np.zeros(self.N)
        ovf_return[:] = self.vmax*np.divide(np.power(Delta_x - 1,3),1+np.power(Delta_x - 1,3))
        
        index = np.where(Delta_x<=1)
        ovf_return[index] = 0
        
        return ovf_return
       
    def steadyStateFlow(self,L,N):
        """
        Returns parameters b, c and f of  a steady state flow.
        Input: 
            L - length of circuit
            N - number of cars
        Returns:
            b - constant spacing
            c - constant velocity
            f - derivative V(b) 
        """
        b = float(L)/float(N)
        c = self.V(b)
        f = 1 - np.tanh(b)**2
        return b,c,f
    
    def density_front(self,x):
        """
        compute the local density for each car
        """
        car_spacing = int(self.n)
        #density front
        box_size = (np.roll(x,-car_spacing) -x)%self.L 
        rho = float(car_spacing)/box_size
        return rho
    
    def density_middle(self,x):
        """
        compute the local density for each car
        """
        car_spacing = int(self.n/2)
        # density middle
        
        box_size = (np.roll(x,-car_spacing) -x)%self.L + (x -np.roll(x,car_spacing))%self.L
        rho = 2*float(car_spacing)/box_size
        return rho
    
    def density_back(self,x):
        """
        compute the local density for each car
        """
        car_spacing = int(self.n)
        #density back
        box_size = (x -np.roll(x,car_spacing))%self.L 
        rho = float(car_spacing)/box_size
        return rho
        
    def flow_velocity_front(self,dotx):
        """
        compute the local flow velocity for each car using moving averages with 
        periodic boundary conditions
        """
        # flow velocity front
        car_spacing = int(self.n)
        dotx_extended = np.append(dotx,dotx[:car_spacing])
        dotx_flow = np.convolve(dotx_extended, np.ones((car_spacing+1,))/(car_spacing+1), mode='valid')
        return dotx_flow
    
    def flow_velocity_middle(self,dotx):
        """
        compute the local flow velocity for each car using moving averages with 
        periodic boundary conditions
        """
        # flow velocity middle
        car_spacing = int(self.n/2)
        dotx_extended = np.append(dotx[-car_spacing:],dotx)
        dotx_extended = np.append(dotx_extended,dotx[:car_spacing])         
        dotx_flow = np.convolve(dotx_extended, np.ones((2*car_spacing+1,))/(2*car_spacing+1), mode='valid')
        return dotx_flow

    def flow_velocity_back(self,dotx):
        """
        compute the local flow velocity for each car using moving averages with 
        periodic boundary conditions
        """
        #flow velocity back
        car_spacing = int(self.n)
        dotx_extended = np.append(dotx[-car_spacing:],dotx)
        dotx_flow = np.convolve(dotx_extended, np.ones((car_spacing+1,))/(car_spacing+1), mode='valid')
        return dotx_flow
    
    def lift_density(self,rho):
        """
        Lifting operator that yields postitions from the local density 
        """
        headway_approx = 1./rho
        position = np.cumsum(headway_approx)
        position = position - position[0]
        return position