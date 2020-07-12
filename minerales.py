import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d




datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
# L = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])
L = 0.01 * (189 - 43.5)/(1-0.02)

L
# N = (1 - datos['NPHI']) / (datos['RHOB'] - 1)
N = (1 - 0.33) / (2.35 - 1)
N

M = (189-DT)/(densidad - 1)


DOLOMIA = np.array([43.5,   2.87,   0.02])
CALIZA  = np.array([47.6,   2.71,   0.00])
SILICE  = np.array([55.5,   2.65,  -0.035])
ARCILLA = np.array([120,    2.35,   0.33])

def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[1] - 1)
    N = (1 - mineral[2]) / (mineral[1] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[2])
    return    np.array([N,M,L])

param_lito(DOLOMIA)
param_lito(CALIZA)
param_lito(SILICE)
param_lito(ARCILLA)

ax = param_lito(DOLOMIA)[0]
ay = param_lito(DOLOMIA)[1]
bx = param_lito(CALIZA)[0]
by = param_lito(CALIZA)[1]
cx = param_lito(SILICE)[0]
cy = param_lito(SILICE)[1]
dx = param_lito(ARCILLA)[0]
dy = param_lito(ARCILLA)[1]
ax,ay
bx,by
cx,cy
dx,dy


P_inicial=[ax,bx,cx,dx,ax]
P_final  =[ay,by,cy,dy,ay]
P_M1=[ax,cx]
P_M2=[ay,cy]
v_x1=[ax,ax]
v_y1=[ay,1.2]
v_x2=[bx,bx]
v_y2=[by,1.2]
v_x3=[cx,cx]
v_y3=[cy,1.2]
