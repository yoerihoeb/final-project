#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:45:59 2018

ovm_init.py contains the initial conditions and parameters for the ovm model
to reproduce differnt papers

@author: paul
"""

import numpy as np

def parameters(argument):
    """
    function that returns the setup dictionary
    """
    switcher = {
        "B95": B95,
        "MS17": MS17,
        "M14":M14,
        "O05":O05,
    }
    return switcher.get(argument, "nothing")

# =============================================================================
# Bando et al (1995)
# =============================================================================
B95 = {
        "N":100,
        "L":200,
        "a":1.,
        "h":2.,
        "tmax":1000,
        "dt" : 0.1,
        "vmax":2,
        "ovf":"tanh",
        "n":5,
        "box":"front",
        "model":"OVM",
        "r":0.0
        }

B95["xpert"] = np.zeros(B95["N"])
B95["xpert"][0] = 0.1

# =============================================================================
# Miura and Sugiyama (2017)
# =============================================================================
MS17 = {
        "N":60,
        "L":60,
        "a":1.7,
        "h":1.2,
        "tmax": 1000,
        "dt" : 0.1,
        "vmax":0.825,
        "ovf":"tanh",
        "n":5,
        "box":"front",
        "model":"OVM"
        }
MS17["xpert"] = np.zeros(MS17["N"])
MS17["xpert"][0] = 0.1
#MS17["xpert"] = 2*np.sin(np.linspace(0,2*np.pi,MS17["N"]))


# =============================================================================
# Marschler et al (2014)
# =============================================================================
M14 = {
        "N":60,
        "L":60,
        "a":1.7,
        "h":1.2,
        "tmax": 50000, # originally 5*10**4
        "dt" : 0.1,
        "vmax":2*0.87, # vmax = 2*v_0 in!!!
        "ovf":"tanh",
        "n":5,
        "box":"front",
        "model":"OVM"
        }
M14["xpert"] = np.zeros(M14["N"])
M14["xpert"] = 0.1*np.sin(2*np.pi/M14["N"]*np.arange(M14["N"]))

# =============================================================================
# Orosz et al (2005)
# =============================================================================
O05 = {
        "N":17,
        "L":60,
        "a":1.7,
        "h":1.2,
        "tmax": 1000, # originally 5*10**4
        "dt" : 0.1,
        "vmax":1., # vmax = 2*v_0 in!!!
        "ovf":"alg",
        "n":5,
        "box":"front",
        "model":"OVM"
        }
O05["xpert"] = np.zeros(M14["N"])
O05["xpert"] = 0.1*np.sin(2*np.pi/M14["N"]*np.arange(M14["N"]))


