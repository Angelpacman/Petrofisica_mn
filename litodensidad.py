import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#datos
datos = pd.read_csv('eval_petro.csv')
datos['DT'] = np.around(np.array( 189 - (datos['RHOB'] -1)*datos['M']/0.01 ),   decimals =4)
datos['N']  = np.around(np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1)),   decimals = 4)
datos['L']  = np.around(np.array( 0.01 * (189-datos['DT']) / (1-datos['NPHI']) ), decimals =4)

#Delimitacion de figura con valores mas precisos de minerales

DOLOMIA = np.array([43.5, 2.87, 0.02])
CALIZA  = np.array([47.6, 2.71, 0.00])
SILICE  = np.array([55.5, 2.65, -0.035])
ARCILLA = np.array([120,  2.35, 0.33])

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

#figura
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
plt.title('M vs N')
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

x = np.around(np.linalg.solve(A, B), decimals = 4)
x
#datos.head()

A_inverse = np.linalg.inv(A)

X = A_inverse * b


FIP  =  X[0]
VDOL =  X[1]
VSIL =  X[2]
VARC =  X[3]


#La idea de poner 2 shape es para que el arreglo quede de tamaño (400,1)
#hasta el momento no he encontrado como optimizar este detalle
FIP = np.array(FIP.T)[0]
FIP.shape
FIP = np.array(FIP.T)[0]
FIP.shape

VDOL = np.array(VDOL.T)[0]
VDOL.shape
VDOL = np.array(VDOL.T)[0]
VDOL.shape

VSIL = np.array(VSIL.T)[0]
VSIL.shape
VSIL = np.array(VSIL.T)[0]
VSIL.shape

VARC = np.array(VARC.T)[0]
VARC.shape
VARC = np.array(VARC.T)[0]
VARC.shape

VCAL = np.array(VARC*0.0000)

datos['VDOL'] = np.around(VDOL,decimals = 4)
datos['VCAL'] = np.around(VCAL,decimals = 4)
datos['VSIL'] = np.around(VSIL,decimals = 4)
datos['VARC'] = np.around(VARC,decimals = 4)
datos['FIP']  = np.around(FIP, decimals = 4)
datos.head()

#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv
