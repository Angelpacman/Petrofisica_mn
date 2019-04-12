
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#datos = pd.read_csv('/home/angelr/Documentos/documentos/evaluacion_petrofisica/litodensidad/eval_petro.csv')
datos = pd.read_csv('eval_petro.csv')

#datos.head()
#datos['PROF'].head()
datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
#datos['DT']
#datos


datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
#datos.head()

datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])
#datos['L']


P_inicial=[0.5051,0.5241,0.5848,0.6273,0.6273,0.5051]
P_final  =[0.702,0.7781,0.8269,0.8091,0.8091,0.702]
P_M1=[0.5241,0.6273]
P_M2=[0.7781,0.8091]
v_x1=[0.5241,0.5241]
v_y1=[0.7781,0.95]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,0.95]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,0.95]


"""
plt.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
plt.plot(datos['N'],datos['M'],marker='o', markersize=2, linestyle='', color='r', label = "M vs N")
#plt.scatter(datos['N'],datos['M'])

#plt.xlim([0.5,0.65])
#plt.ylim([0.7,0.95])
plt.xlim([0.3,1])
plt.ylim([0.4,1.2])
plt.grid()
plt.xlabel('N')
plt.ylabel('M')
plt.title('Gráfica de M vs N')
plt.show()
"""

PROF = np.array(datos['PROF'])
GR = np.array(datos['GR'])
LLS = np.array(datos['LLS'])
FR = np.array(datos['FR'])
DT = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
M = np.array(datos['M'])
N = np.array(datos['N'])
L = np.array(datos['L'])


fig = plt.figure(figsize=(10, 10))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(N, M, marker='o', linestyle='', markersize=7, alpha=0.2, color ='orange')
main_ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
main_ax.grid()


"""plt.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
plt.plot(datos['N'],datos['M'],marker='o', markersize=2, linestyle='', color='r', label = "M vs N")
#plt.scatter(datos['N'],datos['M'])

#plt.xlim([0.5,0.65])
#plt.ylim([0.7,0.95])
plt.xlim([0.3,1])
plt.ylim([0.4,1.2])
plt.grid()
plt.xlabel('N')
plt.ylabel('M')
plt.title('Gráfica de M vs N')"""

# histogram on the attached axes
x_hist.hist(N, 80, histtype='stepfilled',
            orientation='vertical', color='orange')
x_hist.grid()
x_hist.invert_yaxis()


y_hist.hist(M, 80, histtype='stepfilled',
            orientation='horizontal', color='orange')
y_hist.invert_xaxis()
y_hist.grid()


#plt.xlabel('N')
#plt.ylabel('M')


# make some labels invisible
x_hist.xaxis.set_tick_params(labelbottom=False)
y_hist.yaxis.set_tick_params(labelleft=False)


x_hist.set_yticks([0, 15, 30])
y_hist.set_xticks([0, 15, 30])
#datos

plt.show()
