import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap



datos = pd.read_csv("eval_petro.csv")
#datos.head()
#datos['PROF'].head()
datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
#datos['DT']
#datos


datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
#datos.head()

datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])
#datos['L']

phi_primaria1 = [0.5241,0.5848,0.6273,0.5241]
phi_primaria2 = [0.7781,0.8269,0.8091,0.7781]
P_inicial=[0.5051,0.5241,0.5848,0.6273] #,0.6273,0.5051
P_final  =[0.702,0.7781,0.8269,0.8091] #,0.8091,0.702
P_M1=[0.5241,0.6273]
P_M2=[0.7781,0.8091]
v_x1=[0.5241,0.5241]
v_y1=[0.7781,1.2]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,1.2]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,1.2]


#plt.plot(phi_primaria1,phi_primaria2)
#plt.plot(P_inicial,P_final)
#plt.show()

"""
plt.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
plt.plot(datos['N'],datos['M'],marker='.', linestyle='', color='b', label = "M vs N")
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



fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
X = datos['N']
Y = datos['M']
Z = datos['PROF']  #*-1

ax.plot3D(X,Y,Z,color ='orange', marker = 'o')
ax.invert_zaxis()
#Axes3D.plot(xs = datos['N'], ys = datos['M'], zs = datos['L'] , zdir = 'z')a
ax.set_xlabel('N')
ax.set_ylabel('M')
ax.set_zlabel('Z')
#plt.show()




#fig2 = plt.figure()
ay = fig.add_subplot(122, projection='3d')
X = datos['N']
Y = datos['M']
Z = datos['PROF']     #*-1
ay.invert_zaxis()
ay.plot(X,Y,Z, c='blue',marker = 'o', linestyle='')
#Axes3D.plot(xs = datos['N'], ys = datos['M'], zs = datos['L'] , zdir = 'z')a
ay.set_xlabel('N')
ay.set_ylabel('M')
ay.set_zlabel('Z')

plt.show()
