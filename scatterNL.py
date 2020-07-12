
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.mlab as mlab



#lectura de datos y calculo de las variables que faltan
datos = pd.read_csv('eval_petro(0).csv')

# datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
# datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
# datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])
datos['M'] = np.array( 0.01 * (189-datos['DT'])/(datos['RHOB'] - 1) )
datos['N'] = np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1) )
datos['L'] = np.array( 0.01 * (189 - datos['DT'])/(1 - datos['NPHI']) )


#pares de puntos para gradicar el triangulo y las lineas de porosidad primaria
#pares de puntos para graficar el triangulo y las lineas de porosidad primaria
#Arreglos que contienen DT, RHOB y NPHI de cada mineral
DOLOMIA = np.array([43.5,   2.87,   0.02])
CALIZA  = np.array([47.6,   2.71,   0.00])
SILICE  = np.array([55.5,   2.65,  -0.035])
ARCILLA = np.array([120,    2.35,   0.33])

#funcion para calcular parametros
def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[1] - 1)
    N = (1 - mineral[2]) / (mineral[1] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[2])
    return    np.array([M,N,L])


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



"""Aqui hemos de convertir las colimnas del dataframe que salio de pandas para transformarlas en
variables que se pueden manipular como arreglos"""
PROF =  np.array(datos['PROF'])
GR   =  np.array(datos['GR'])
LLS  =  np.array(datos['LLS'])
FR   =  np.array(datos['FR'])
DT   =  np.array(datos['DT'])
NPHI =  np.array(datos['NPHI'])
M    =  np.array(datos['M'])
N    =  np.array(datos['N'])
L    =  np.array(datos['L'])

#ubucacion de la figura y las subplots
fig     = plt.figure(figsize=(9, 9))
grid    = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[1:,:-1])
y_hist  = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
x_hist  = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(N, L, marker='o', linestyle='', markersize=4, alpha=0.3, color ='orange')
main_ax.plot(P_inicial,P_final)
main_ax.grid()



# histogram on the attached axes
x_hist.hist(N, 150, histtype='stepfilled',orientation='vertical', color='orange')
x_hist.grid()
#x_hist.invert_yaxis()

y_hist.hist(L, 150, histtype='stepfilled',orientation='horizontal', color='orange')
#y_hist.invert_xaxis()
y_hist.grid()



# make some labels invisible
x_hist.xaxis.set_tick_params(labelbottom=False)
y_hist.yaxis.set_tick_params(labelleft=False)

#niveles de grid en las subplot pequeñas (frecuancia de datos)
x_hist.set_yticks([0, 5, 10, 15])
y_hist.set_xticks([0, 5, 10, 15])

x_hist.set_ylabel('N frequency')
main_ax.set_xlabel('N')
main_ax.set_ylabel('L')
y_hist.set_xlabel('L frequency')

x_hist.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
y_hist.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')



plt.show()
