#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Rafael Figueroa
Hybrid Automata Model and simulation using HaSimPy
"""

from robot_vars import *
import numpy as np
from hasimpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sin = np.sin
cos = np.cos
tan = np.tan
pi = np.pi

def u0(X = None):
    """Passive Walking"""
    return 0.0

def f0(X, u = u0(), t=0):
    """Compass Gait Dynamics"""

    hsw = X[0]
    hst = X[1]
    hdsw = X[2]
    hdst = X[3]
    u = u()

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

    # print 'reset map Xold:', X

    Pl = [[-a*l*m*cos(1.0*hst - hsw)/(M*l**2 + a**2*m - l**2*m*cos(1.0*hst - \
        hsw)**2 + l**2*m), (-M*a*l**2 + M*l**3*cos(1.0*hst - hsw)**2 - \
        a**3*m + 2*a*l**2*m*cos(1.0*hst - hsw)**2 - a*l**2*m)/(b*(M*l**2 + \
        a**2*m - l**2*m*cos(1.0*hst - hsw)**2 + l**2*m))], \
        [-a*b*m/(M*l**2 + a**2*m - l**2*m*cos(1.0*hst - hsw)**2 + l**2*m), \
        l*(M*l + a*m)*cos(1.0*hst - hsw)/(M*l**2 + a**2*m - \
        l**2*m*cos(1.0*hst - hsw)**2 + l**2*m)]]

    P = np.array(Pl)
    J = np.array([[0, 1], [1, 0]])

    # print 'reset map P:', P

    # q = [hsw, hst] transposed
    # dq = [hdsw, hdst] transposed
    # q = J*q (before and after)
    # dq = P*dq (before and after)
    
    q_old = np.array([hsw, hst])
    dq_old = np.array([hdsw, hdst])

    q = np.dot(J, q_old.transpose())
    dq = np.dot(P, dq_old.transpose())

    Xnew = np.hstack((q, dq))
    # print 'reset map Xnew:', Xnew

    return Xnew

def g0(X):
    """Guard: activates (True) when the swing leg impacts the ground"""
    hsw = X[0]
    hst = X[1]
   
    impact_cond = tolEqual(tan(hst/2 + hsw/2), tan(pi-gamma))
    walking_cond = (hsw > hst)
    # print 'hs',tan(hst/2 + hsw/2), 
    # 'gm', tan(pi-gamma), 'imp', impact_cond, 'wal', walking_cond
    
    return impact_cond and walking_cond

def avoid(X):
    """No avoid set analysis for this system"""
    return False  # Never goes into the avoid set


e0 = E([0], [g0], [r0])
q0 = Q(0, f0, u0, e0, Dom=any, Avoid=avoid)
# [0.086, 0.086, -1.34 + (-0.38+1.34)/2.0, 0.58+(1.54-0.58)*(3/4)]
# [0.5, -0.3, -3, -2]
# [ 0.12402108 -0.26131208 -1.11743065 -0.82005916]
init_X = np.array([-0.22402108, 0.16131208, 0.6, -1.0])
init_qID = 0

states = ['\theta_{sw}', '\theta_{st}', 
          '\dot{\theta}_{sw}', '\dot{\theta}_{st}']

h = H([q0], Init_X = init_X, Init_qID = init_qID, 
      state_names = states )

simResult = h.sim(qID = 0, X = init_X, u = u0, t0 = 0, tlim = 3, 
                  debug_flag = False)

simPath = simResult.path

def show_plots():
    simResult.simPlot()
    simResult.phasePlot([0, 2])
    simResult.phasePlot([1, 3])

    raw_input('\n Press ENTER to close plots')
def point0(hst, hsw):
    """Standing leg foot position and origin"""
    p0x = 0
    p0x = 0
    return np.array([p0x, p0x])

def point3(hst, hsw):
    """Hip position"""
    p3x = -l*sin(hst)
    p3y =  l*cos(hst)
    return np.array([p3x, p3y])

def point4(hst, hsw):
    """Swing leg foot position"""
    p4x = l*(sin(hsw)-sin(hst))
    p4y = l*(cos(hst)-cos(hsw))
    return np.array([p4x, p4y])

def update_line(frame_id):
    """Updates leg positions on existing animation plot"""
    sim_X = simPath[frame_id]

    hsw = sim_X[0]
    hst = sim_X[1]

    p_st = point0(hst, hsw)
    p_hp = point3(hst, hsw)
    p_sw = point4(hst, hsw)
    
    # First for standing leg
    leg1.set_data([[p_st[0], p_hp[0]], [p_st[1], p_hp[1]]])
    leg2.set_data([[p_hp[0], p_sw[0]], [p_hp[1], p_sw[1]]])

    return [leg1, leg2] 

def animation_init():
    """Clears the lines from the screen"""
    leg1.set_data([], [])
    leg2.set_data([], [])
    return [leg1, leg2]

def animate_walking():
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.title('leg test')

    global leg1, leg2, ground
    fig1 = plt.figure()
    leg1, = plt.plot([], [], 'r-', lw=3)
    leg2, = plt.plot([], [], 'b-', lw=3)
    ground, = plt.plot([- 2*l*cos(gamma),
                        2*l*cos(gamma)], 
                        [2*l*sin(gamma), 
                        -2*l*sin(gamma)], 'k-', lw=5)


    # interval in ms, Ts in seconds
    # running 100 times faster
    legs_ani = animation.FuncAnimation(fig1, update_line, len(simPath),
                                       init_func = animation_init,
                                       interval = 0.0, 
                                       blit = True,
                                       repeat = False)

    plt.show()

    raw_input('\n Press ENTER to close plots')



