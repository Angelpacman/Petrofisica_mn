import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


#captura de los datos
datos = pd.read_csv("eval_petro.csv")
datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])

#definicion de los puntos que van a ser los contenedores del poligono
phi_primaria1 = [0.5241,0.5848,0.6273,0.5241]
phi_primaria2 = [0.7781,0.8269,0.8091,0.7781]
P_inicial=[0.5051,0.5241,0.5848,0.6273,0.6273,0.5051]
P_final  =[0.7020,0.7781,0.8269,0.8091,0.8091,0.7020]
P_M1=[0.5241,0.6273]
P_M2=[0.7781,0.8091]
v_x1=[0.5241,0.5241]
v_y1=[0.7781,0.95]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,0.95]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,0.95]



#convertir la columnas del dataframe en arreglos para poder manipular los datos
N = np.array(datos['N'])
M = np.array(datos['M'])
PROF= np.array(datos['PROF'])  #*-1
col = np.linspace(PROF[-1],PROF[0],400)

#aqui va figura en 2D
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
ax.grid()
ax.set_xlabel('N')
ax.set_ylabel('M')
ax.scatter(N, M, s=10, c=col, marker='o')



#aqui va la figura pero en proyeccion 3D de M vs N
fig = plt.figure()
ay = fig.add_subplot(111, projection='3d')
p3d=ay.scatter(N, M, PROF, s=40, c=col, marker='.')
ay.invert_zaxis()
ay.set_xlabel('N')
ay.set_ylabel('M')
ay.set_zlabel('z')
#plt.colorbar(p3d)
plt.show()
