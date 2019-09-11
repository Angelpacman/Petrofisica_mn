
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.mlab as mlab



#lectura de datos y calculo de las variables que faltan
datos = pd.read_csv('eval_petro.csv')
datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])

#pares de puntos para graficar el triangulo y las lineas de porosidad primaria
#Arreglos que contienen DT, RHOB y NPHI de cada mineral
DOLOMIA = np.array([43.5, 2.87, 0.02])
CALIZA  = np.array([47.6, 2.71, 0.00])
SILICE  = np.array([55.5, 2.65, -0.035])
ARCILLA = np.array([120,  2.35, 0.33])

#funcion para calcular parametros
def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[1] - 1)
    N = (1 - mineral[2]) / (mineral[1] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[2])
    return    np.array([M,N,L])

param_lito(DOLOMIA)
param_lito(CALIZA)
param_lito(SILICE)
param_lito(ARCILLA)

ax = param_lito(DOLOMIA)[1]
ay = param_lito(DOLOMIA)[0]
bx = param_lito(CALIZA)[1]
by = param_lito(CALIZA)[0]
cx = param_lito(SILICE)[1]
cy = param_lito(SILICE)[0]
dx = param_lito(ARCILLA)[1]
dy = param_lito(ARCILLA)[0]

#pares de arreglos x,y para graficar las lineas de la grafica
P_inicial=[ax,bx,cx,dx,ax]
P_final  =[ay,by,cy,dy,ay]
P_M1=[ax,cx]
P_M2=[ay,cy]
v_x1=[ax,ax]
v_y1=[ay,1]
v_x2=[bx,bx]
v_y2=[by,1]
v_x3=[cx,cx]
v_y3=[cy,1]

"""Aqui hemos de convertir las colimnas del dataframe que salio de pandas para transformarlas en
variables que se pueden manipular como arreglos"""
PROF = np.array(datos['PROF'])
GR = np.array(datos['GR'])
LLS = np.array(datos['LLS'])
FR = np.array(datos['FR'])
DT = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
M = np.array(datos['M'])
N = np.array(datos['N'])
L = np.array(datos['L'])

#ubucacion de la figura y las subplots
fig = plt.figure(figsize=(9, 9))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[1:,:-1])
y_hist = fig.add_subplot(grid[1:, -1], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[0, :-1], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(N, M, marker='o', linestyle='', markersize=4, alpha=0.3, color ='orange')
main_ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
main_ax.grid()



# histogram on the attached axes
x_hist.hist(N, 150, histtype='stepfilled',orientation='vertical', color='orange')
x_hist.grid()
#x_hist.invert_yaxis()

y_hist.hist(M, 150, histtype='stepfilled',orientation='horizontal', color='orange')
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
main_ax.set_ylabel('M')
y_hist.set_xlabel('M frequency')

x_hist.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
y_hist.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')



plt.show()
