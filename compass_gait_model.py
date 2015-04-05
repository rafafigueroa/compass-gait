#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Rafael Figueroa
Hybrid Automata Model and simulation using HaSimPy
"""

from robot_vars import *
import numpy as np
from hasimpy import *

sin = np.sin
cos = np.cos
pi = np.pi

def u0(X = None):
    """Passive Walking"""
    return 0

def f0(X, u = u0(), t=0):
    """Compass Gait Dynamics"""

    hsw = X[0]
    hst = X[1]
    hdsw = X[2]
    hdst = X[3]

    dx1 = hdsw
    dx2 = hdst

    dx3  = (b*l*m*(b*hdsw**2*l*m*sin(hst - hsw) + \
           g*(M*l + a*m + l*m)*sin(hst) - 1.0*u)*cos(hst - hsw) - \
           (a**2*m + l**2*(M + m))*(b*g*m*sin(hsw) + \
           b*hdst**2*l*m*sin(hst - hsw) - 1.0*u))/(b**2*m*(a**2*m - \
           l**2*m*cos(hst - hsw)**2 + l**2*(M + m)))

    dx4 = (b*(b*hdsw**2*l*m*sin(hst - hsw) + g*(M*l + a*m + l*m)*sin(hst) - \
          1.0*u) - l*(b*g*m*sin(hsw) + b*hdst**2*l*m*sin(hst - hsw) - \
          1.0*u)*cos(hst - hsw))/(b*(a**2*m - l**2*m*cos(hst - hsw)**2 +  \
          l**2*(M + m)))

    dX = np.array([dx1, dx2, dx3, dx4])

    return dX 

def r0(X):
    """Reset Map at impact"""

    hsw = X[0]
    hst = X[1]
    hdsw = X[2]
    hdst = X[3]

    Wl = [[-a*l*m*cos(1.0*hst - hsw)/(M*l**2 + a**2*m - l**2*m*cos(1.0*hst - \
        hsw)**2 + l**2*m), (-M*a*l**2 + M*l**3*cos(1.0*hst - hsw)**2 - \
        a**3*m + 2*a*l**2*m*cos(1.0*hst - hsw)**2 - a*l**2*m)/(b*(M*l**2 + \
        a**2*m - l**2*m*cos(1.0*hst - hsw)**2 + l**2*m))], \
        [-a*b*m/(M*l**2 + a**2*m - l**2*m*cos(1.0*hst - hsw)**2 + l**2*m), \
        l*(M*l + a*m)*cos(1.0*hst - hsw)/(M*l**2 + a**2*m - \
        l**2*m*cos(1.0*hst - hsw)**2 + l**2*m)]]

    W = np.array(Wl)
    J = np.array([[0, 1], [1, 0]])

    # q = [hsw, hst] transposed
    # dq = [hdsw, hdst] transposed
    # q = J*q (before and after)
    # dq = W*dq (before and after)

    q = np.dot(J, np.array([[hsw], [hst]]))
    dq = np.dot(W, np.array([[hdsw],[hdst]]))

    Xnew = np.vstack((q, dq))

    return Xnew

def g0(X):
    """Guard: activates (True) when the swing leg impacts the ground"""
    return tolEqual(hst+hsw+2*gamma, 0)



