#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Rafael Figueroa
"""
from __future__ import division
import numpy as np
from sympy import *
import shelve
init_printing()

m, l, g, mh, a, b = symbols('m, l, g, mh, a, b')
hst, hsw, hdst, hdsw = symbols('hst, hsw, hdst, hdsw')
u = symbols('u')

H00 = m*b**2
H01 = -m*l*b*cos(hst-hsw)
H10 = -m*l*b*cos(hst-hsw)
H11 = (mh+m)*l**2+m*a**2

C00 = 0
C01 = m*l*b*sin(hst-hsw)*hdst
C10 = m*l*b*sin(hst-hsw)*hdsw
C11 = 0

G00 = m*b*g*sin(hsw)
G10 = -(mh*l+m*a+m*l)*g*sin(hst)

B00 = 1.0
B10 = -1.0

H = Matrix([[H00, H01],[H10, H11]])
C = Matrix([[C00, C01],[C10, C11]])
G = Matrix(2, 1, [G00, G10])
B = Matrix(2, 1, [B00, B10])

qd = Matrix(2, 1, [hdsw, hdst])

print ('H')
pprint(H)
print ('C')
pprint(C)
print ('G')
pprint(G)
print ('B')
pprint(B)
print('qd')
pprint(qd)
print ('C*qd')
pprint(C*qd)
print ('B*u')
pprint(B*u)

qdd = H.inv()*(-C*qd -G + B*u)
print('qdd')
pprint(qdd)

qdd0 = qdd[0]
qdd1 = qdd[1]
print('qdd 0 and 1')
pprint(qdd0)
pprint(qdd1)


