import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#datos
datos = pd.read_csv('eval_petro.csv')
datos['DT'] = np.around(np.array( 189 - (datos['RHOB'] -1)*datos['M']/0.01 ),  decimals =4)
datos['N']  = np.around(np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1)), decimals = 4)
datos['L']  = np.around(np.array( 0.01 * (189 - datos['DT']) / (1-datos['NPHI']) ) , decimals =4)

#Delimitacion de figura
P_inicial=[0.5051,0.5241,0.5848,0.6273,0.6273,0.5051]
P_final  =[0.702,0.7781,0.8269,0.8091,0.8091,0.702]
P_M1=[0.5241,0.6273]
P_M2=[0.7781,0.8091]
v_x1=[0.5241,0.5241]
v_y1=[0.7781,1.2]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,1.2]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,1.2]
#print(P_inicial)
#plt.plot(P_inicial,P_final)
#plt.plot(P_M1,P_M2)


#figura
plt.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
plt.plot(datos['N'],datos['M'],marker='o', markersize=1, linestyle='', color='r', label = "M vs N")
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


#librerias para trabajar con el poligono
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#vertices del poligono
polygon = Polygon([(0.5241, 0.7781), (0.5848, 0.8269), (0.6273, 0.8091), (0.5241, 0.7781)])


"""definicion de los datos, conversion de datos de serie a numericos tipo array"""
M = np.array(datos['M'])
N = np.array(datos['N'])

#algoritmo para decidir si un punto esta dentro del poligono
i = 0
puntos = []
for numero in M:
    point = Point(N[i],M[i])
    buleano = polygon.contains(point)
    if buleano == True:
        puntos.append('primaria')
    else:
        puntos.append('secundaria')
    i += 1

#no recuerdo por que meti denuevo los datos ¿¿?¿?¿'¿¿'??? xdxdxd
datos['Porosidad'] = puntos
datos.to_csv('eval_petro_output.csv') #ahhh era para sacar un excel diferente


#manipulacion de datos como arreglos
PROF = np.array(datos['PROF'])
GR   = np.array(datos['GR'])
LLS  = np.array(datos['LLS'])
FR   = np.array(datos['FR'])
DT   = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
RHOB = np.array(datos['RHOB'])
M    = np.array(datos['M'])
N    = np.array(datos['N'])
L    = np.array(datos['L'])
Porosidad = np.array(datos['Porosidad'])
#print(datos)
datos.head()
print(datos)
plt.plot(GR,PROF,FR,PROF)
plt.grid()
plt.show()
np.mean(GR)


#gr 0 150
#resistivos 0.2 2000
#dt 45 189

#prube de ploteo con registros xdxdxdddd
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.01, 5.0, 0.01)
s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = np.sin(4 * np.pi * t)

ay1 = plt.subplot(131)
plt.xlim([0,70])
plt.plot(GR, PROF)
#plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ay2 = plt.subplot(132, sharex=ay1)
plt.xlim([0,150])
plt.plot(DT, PROF)

# make these tick labels invisible
#plt.setp(ax2.get_xticklabels(), visible=False)

# share x and y
ay3 = plt.subplot(133, sharex=ay1, sharey=ay1)
plt.plot(FR, PROF)
#plt.xlim(0.01, 5.0)
plt.show()


"""
ahora toca armar el sistema de ecuaciones y resolverlo para todas las filas
"""
# define matrix A using Numpy arrays
A = np.matrix([ [189, 43.5, 55.5,   120],
                [1.0, 0.02, -0.035, 0.33],
                [1.0, 2.87, 2.65,   2.35],
                [1.0, 1.0,  1.0,    1.0]    ])
A.shape

#define matrix B
b = np.matrix([ [DT],
                [NPHI],
                [RHOB],
                [1] ])
# b = np.array([73.9477, 0.1275, 2.6503, 1])
B = np.array([59.8739, 0.0606, 2.5407, 1])

x = np.around(np.linalg.solve(A, B), decimals = 7)
x
datos.head()

A_inverse = np.linalg.inv(A)

X = A_inverse * b


FIP  =  X[0]
VDOL =  X[1]
VSIL =  X[2]
VARC =  X[3]


FIP = np.array(FIP.T)[0]
type(FIP)
FIP.shape
FIP.size

VDOL = np.array(VDOL.T)[0]
type(VDOL)
VDOL.shape
VDOL.size

VSIL = np.array(VSIL.T)[0]
type(VSIL)
VSIL.shape
VSIL.size

VARC = np.array(VARC.T)[0]
type(VARC)
VARC.shape
VARC.size

datos['FIP']  = np.around(FIP,  decimals = 4)
datos['VDOL'] = np.around(VDOL, decimals = 4)
datos['VSIL'] = np.around(VSIL, decimals = 4)
datos['VARC'] = np.around(VARC, decimals = 4)
datos.head()
