#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Rafael Figueroa
Stochastic Gradient Descent
"""

from __future__ import division
from robot_vars import *
import numpy as np
from hasimpy import *
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import compass_gait_model
from compass_gait_model import h

sin = np.sin
cos = np.cos
tan = np.tan
pi = np.pi

# print all the array
np.set_printoptions(threshold=np.inf)

def normal_noise(sigma, states):
    """ Draw normal noise """
    return np.random.normal(0, sigma, states)   

def poincare_map_compass(path):
    """ Records the crossing values of the surface of section """
    hst_path = path[:, 1]
    hdst_path = path[:, 3]
    hst_sign = np.sign(hst_path)
    hst_zeros = np.where(hst_sign == 0)[0]
    # Remove the zeros in all arrays to 
    # mantain consistency of index
    np.delete(hst_sign, hst_zeros)
    np.delete(hst_path, hst_zeros)
    np.delete(hdst_path, hst_zeros)

    # Check for sign change, but only from positive to negative
    # defining the surface of section as hst = 0 when it
    # crosses from positive to negative
    hst_sign_change = ((hst_sign - np.roll(hst_sign, -1)) > 0).astype(int)
    # np.roll compares the last with the first, so the last  result
    # is not meaningless in this problem
    hst_sign_change[-1] = 0

    # These are the values at the surface of section crossing
    # Just before it crosses
    hst_poincare_vals = hst_path[np.nonzero(hst_sign_change)]
    # print "hst_poincare_vals", hst_poincare_vals
    hdst_poincare_vals = hdst_path[np.nonzero(hst_sign_change)]
    # print "hdst_poincare_vals", hdst_poincare_vals

    return hdst_poincare_vals
    
def J(a):
    """ Cost Function """
    u0 = compass_gait_model.u0
    simResult = h.sim(qID = 0, X = a, u = u0, t0 = 0, tlim = 3, 
                  debug_flag = False, Ts = 1e-3)

    simPath = simResult.path
    poincare_vals = poincare_map_compass(simPath)

    # The cost function is the difference between them
    if len(poincare_vals) >= 3:
        J_all = np.linalg.norm(np.roll(poincare_vals, 1) - poincare_vals) 
        J_last = np.abs(poincare_vals[-1] - poincare_vals[-2])
        J_first = np.abs(poincare_vals[0] - poincare_vals[1])
        return J_first + J_all

    else:
        # penalize taking less steps!
        return 1

def run_sgd(a0, sigma = 0.1, eta = 0.004, states = 4):
    # Parameters
    a = a0
    # Get cost baseline
    Jb = J(a)
    Jmin = Jb
    converged = False
    J_list = []
    print Jb, eta, a 

    while not converged:
        Z = normal_noise(sigma, states)
        # Explore cost on new direction
        Je = J(a + Z)
        # Update search direction
        Da = -eta * (Je - Jb) * Z
        # Update parameters
        a += Da
        # Update Baseline
        Jb = J(a)

        # J_list.append(Jb)
        eta = eta/1.02
        sigma = sigma/1.05

        # print Jb, eta, a 
        if Jb < Jmin:
            Jmin = Jb
            print Jb, eta, a 
        
        # Evaluate converging condition
        if Jb < 0.001:
            converged = True
            return a, Jb


if __name__ == '__main__': 
    # [-0.22402108, 0.16131208, 0.6, -1.0 ]
    a0 = np.array([-0.22, 0.16, 0.6, -1.0])
    a, Jmin = run_sgd(a0)
    print a, Jmin



        

    
    




    
