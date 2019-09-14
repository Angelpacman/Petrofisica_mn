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

DOLOMIA = np.array([43.5, 0.02, 2.87])
CALIZA  = np.array([47.6, 0.00, 2.71])
SILICE  = np.array([55.5,-0.035, 2.65])
ARCILLA = np.array([120,  0.33,  2.35])

def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[2] - 1)
    N = (1 - mineral[1]) / (mineral[2] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[1])
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
r1 = DOL_CAL_SIL_FIP = Polygon([(ax, ay), (bx , by), (cx , cy), (ax, ay)])
r2 = DOL_SIL_ARC_FIP = Polygon([(ax, ay), (cx , cy), (dx , dy), (ax, ay)])
r3 = DOL_CAL_FIP_FIS = Polygon([(ax, ay), (bx , by), (bx , 1.2), (ax, 1.2), (ax ,ay)])
r4 = CAL_SIL_FIP_FIS = Polygon([(bx, by), (cx , cy), (cx , 1.2), (bx ,1.2), (bx, by)])


"""definicion de los datos, conversion de datos de serie a numericos tipo array"""
M = np.array(datos['M'])
N = np.array(datos['N'])

#algoritmo para decidir si un punto esta dentro del poligono
i = 0
puntos = []

for numero in M:
    point = Point(N[i],M[i])
    buleano = DOL_CAL_SIL_FIP.contains(point)

    if buleano == True:
        puntos.append('DOL_CAL_SIL_FIP')

    else:
        buleano = DOL_SIL_ARC_FIP.contains(point)
        if buleano == True:
            puntos.append('DOL_SIL_ARC_FIP')

        else:
            buleano = DOL_CAL_FIP_FIS.contains(point)
            if buleano == True:
                puntos.append('DOL_CAL_FIP_FIS')

            else:
                buleano = CAL_SIL_FIP_FIS.contains(point)
                if buleano == True:
                    puntos.append('CAL_SIL_FIP_FIS')

                else:
                    puntos.append('null')
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


#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv
r1 = str("DOL_CAL_SIL_FIP")
r2 = str("DOL_SIL_ARC_FIP")
r3 = str("DOL_CAL_FIP_FIS")
r4 = str("CAL_SIL_FIP_FIS")




FIP = np.array([])
VDOL = np.array([])
VCAL = np.array([])
VSIL = np.array([])
VARC = np.array([])
FIS = np.array([])


for area in Porosidad:

    i = 0
    if area == r1:
        A = np.array([ [189, 43.5, 47.5,   55.5],
                        [1.0, 0.02, 0.0, -0.035],
                        [1.0, 2.87, 2.71,   2.65],
                        [1.0, 1.0,  1.0,    1.0]    ])
        A.shape
        #define matrix B
        b = np.array([ DT[i],
                        NPHI[i],
                        RHOB[i],
                        1 ])
        X    = np.linalg.solve(A, b)
        FIP  = np.append(FIP, [X[0]], axis=0)
        VDOL = np.append(VDOL,[X[1]], axis=0)
        VSIL = np.append(VSIL,[X[3]], axis=0)
        VARC = np.append(VARC,[0],    axis=0)
        VCAL = np.append(VCAL,[X[2]], axis=0)
        FIS  = np.append(FIS, [0],    axis=0)



    else:
        if area == r2:
            A = np.array([ [189, 43.5, 55.5,   120],
                            [1.0, 0.02, -0.035, 0.33],
                            [1.0, 2.87, 2.65,   2.35],
                            [1.0, 1.0,  1.0,    1.0]    ])
            A.shape
            #define matrix B
            b = np.array([ DT[i],
                            NPHI[i],
                            RHOB[i],
                            1 ])
            X    = np.linalg.solve(A, b)
            FIP  = np.append(FIP, [X[0]], axis=0)
            VDOL = np.append(VDOL,[X[1]], axis=0)
            VSIL = np.append(VSIL,[X[2]], axis=0)
            VARC = np.append(VARC,[X[3]], axis=0)
            VCAL = np.append(VCAL,[0],    axis=0)
            FIS  = np.append(FIS, [0],    axis=0)



        else:
            if area == r3:
                A = np.matrix([ [189, 45.55, 43.5, 47.6],
                                [1.0, 1.0, 0.02,    0.0],
                                [1.0, 1.0, 2.87,   2.71],
                                [1.0, 1.0,  1.0,    1.0]    ])
                A.shape
                #define matrix B
                b = np.array([ DT[i],
                                NPHI[i],
                                RHOB[i],
                                1 ])
                X    = np.linalg.solve(A, b)
                FIP  = np.append(FIP, [X[0]], axis=0)
                VDOL = np.append(VDOL,[X[2]], axis=0)
                VSIL = np.append(VSIL,[0],    axis=0)
                VARC = np.append(VARC,[0],    axis=0)
                VCAL = np.append(VCAL,[X[3]], axis=0)
                FIS  = np.append(FIS, [X[1]], axis=0)



            else:
                if area == r4:
                    A = np.array([  [189, 51.55, 47.6, 55.5],
                                    [1.0, 1.0,  0.0, -0.035],
                                    [1.0, 1.0, 2.71,   2.65],
                                    [1.0, 1.0,  1.0,    1.0]    ])
                    A.shape
                    #define matrix B
                    b = np.array([ DT[i],
                                    NPHI[i],
                                    RHOB[i],
                                    1 ])
                    X    = np.linalg.solve(A, b)
                    FIP  = np.append(FIP, [X[0]], axis=0)
                    VDOL = np.append(VDOL,[0],    axis=0)
                    VSIL = np.append(VSIL,[X[3]], axis=0)
                    VARC = np.append(VARC,[0],    axis=0)
                    VCAL = np.append(VCAL,[X[2]], axis=0)
                    FIS  = np.append(FIS, [X[1]], axis=0)


                else:
                    FIP  = np.append(FIP, [np.NaN], axis=0)
                    VDOL = np.append(VDOL,[np.NaN], axis=0)
                    VSIL = np.append(VSIL,[np.NaN], axis=0)
                    VARC = np.append(VARC,[np.NaN], axis=0)
                    VCAL = np.append(VCAL,[np.NaN], axis=0)
                    FIS  = np.append(FIS, [np.NaN], axis=0)
    i += 1

FIS.shape
datos['VDOL'] = np.around(VDOL,decimals = 4)
datos['VCAL'] = np.around(VCAL,decimals = 4)
datos['VSIL'] = np.around(VSIL,decimals = 4)
datos['VARC'] = np.around(VARC,decimals = 4)
datos['FIS']  = np.around(FIS, decimals = 4)
datos['FIP']  = np.around(FIP, decimals = 4)

datos.to_csv('eval_petro_output.csv') #exportando al archivo csv

VDOL[56]+ VCAL[56] +VSIL[56]+VARC[56]+FIS[56]+FIP[56]
#plt.plot(VSIL,-1*PROF)
print(datos)
