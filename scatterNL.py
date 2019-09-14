
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

#pares de puntos para gradicar el triangulo y las lineas de porosidad primaria
#P_inicial=[0.5051,0.5241,0.5848,0.6273,0.6273,0.5051]
#P_inicial  =[0.7781,0.8269,0.8091,0.7781]
P_inicial =[0.5241,0.5848, 0.6273, 0.5241]
P_final = [1.4847, 1.4140, 1.2898, 1.4847]
P_M1=[0.5241,0.6273]
P_M2=[0.7781,0.8091]
v_x1=[0.5241,0.5241]
v_y1=[0.7781,0.95]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,0.95]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,0.95]


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
