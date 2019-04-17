import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

datos = pd.read_csv('eval_petro.csv')
datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])

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



PROF = np.array(datos['PROF']) #*-1 #esto para poner profundidad negativa, ojo: invertir zaxis()
GR = np.array(datos['GR'])
LLS = np.array(datos['LLS'])
FR = np.array(datos['FR'])
DT = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
M = np.array(datos['M'])
N = np.array(datos['N'])
L = np.array(datos['L'])



# Create plot 3D Plot
fig = plt.figure()
#Aqui precisamos de un arreglo que tenga el mismo tamaño de M y N (que en nuestro caso es 400)
#col = np.arange(np.array(len(PROF)))
#col = np.linspace(PROF[0],PROF[399],400)
col = np.linspace(PROF[-1],PROF[0],400)

ax3D = fig.add_subplot(111, projection='3d')
p3d=ax3D.scatter(N, M, PROF, s=100, c=col, marker='.')
ax3D.invert_zaxis()
ax3D.set_xlabel('N')
ax3D.set_ylabel('M')
ax3D.set_zlabel('z')
#plt.colorbar(p3d)
plt.show()
