import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d



#captura de los datos
datos = pd.read_csv("eval_petro.csv")
# datos = pd.read_csv("gabs.csv")
#datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
#datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
#datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])

M = np.array( 0.01 * (189-datos['DT'])/(datos['RHOB'] - 1) )
N = np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1) )
L = np.array( 0.01 * (189 - datos['DT'])/(1 - datos['NPHI']) )
datos['M'] = np.around(M, decimals = 4)
datos['N'] = np.around(N, decimals = 4)
datos['L'] = np.around(L, decimals = 4)
#datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
# L = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])
L = 0.01 * (189 - 43.5)/(1-0.02)

L
# N = (1 - datos['NPHI']) / (datos['RHOB'] - 1)
N = (1 - 0.33) / (2.35 - 1)
N
"""
Formula de M
M = (189-DT)/(densidad - 1)
"""


DOLOMIA = np.array([43.5,   2.87,   0.02])
CALIZA  = np.array([47.6,   2.71,   0.00])
SILICE  = np.array([55.5,   2.65,  -0.035])
ARCILLA = np.array([120,    2.35,   0.33])

def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[1] - 1)
    N = (1 - mineral[2]) / (mineral[1] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[2])
    return    np.array([N,M,L])

a_x = param_lito(DOLOMIA)[0]
a_y = param_lito(DOLOMIA)[1]
a_z = param_lito(DOLOMIA)[2]

b_x = param_lito(CALIZA)[0]
b_y = param_lito(CALIZA)[1]
b_z = param_lito(CALIZA)[2]

c_x = param_lito(SILICE)[0]
c_y = param_lito(SILICE)[1]
c_z = param_lito(SILICE)[2]

d_x = param_lito(ARCILLA)[0]
d_y = param_lito(ARCILLA)[1]
d_z = param_lito(ARCILLA)[2]

P_inicial=[a_x, b_x,    c_x,    d_x,    a_x]
P_final  =[a_y, b_y,    c_y,    d_y,    a_y]
P_M1=[a_x,c_x]
P_M2=[a_y,c_y]
v_x1=[a_x,a_x]
v_y1=[a_y,1]
v_x2=[b_x,b_x]
v_y2=[b_y,1]
v_x3=[c_x,c_x]
v_y3=[c_y,1]

tirang_dol_cal_sil_A = [a_x,    b_x,    c_x,    a_x]
tirang_dol_cal_sil_B = [a_y,    b_y,    c_y,    a_y]
tirang_dol_cal_sil_C = [a_z,    b_z,    c_z,    a_z]


triang_dol_sil_arc_A = [a_x,    c_x,    d_x,    a_x]
triang_dol_sil_arc_B = [a_y,    c_y,    d_y,    a_y]
triang_dol_sil_arc_C = [a_z,    c_z,    d_z,    a_z]
