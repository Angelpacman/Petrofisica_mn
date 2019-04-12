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
v_y1=[0.7781,1.2]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,1.2]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,1.2]



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
phi_primaria1 = [0.5241,0.5848,0.6273,0.5241]
phi_primaria2 = [0.7781,0.8269,0.8091,0.7781]
plt.plot(phi_primaria1,phi_primaria2)
"""


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#point = Point(0.5848, 0.8)
polygon = Polygon([(0.5241, 0.7781), (0.5848, 0.8269), (0.6273, 0.8091), (0.5241, 0.7781)])
#print(polygon.contains(point))

"""definicion de los datos, conversio de datos de serie a numericos tipo array"""
#M = pd.to_numeric(datos['M'])
M = np.array(datos['M'])
#M = np.array(pd.to_numeric(datos['M']))
#N = pd.to_numeric(datos['N'])
N = np.array(datos['N'])
PROF = np.array(datos['PROF'])
GR = np.array(datos['GR'])
LLS = np.array(datos['LLS'])
FR = np.array(datos['FR'])
DT = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
M = np.array(datos['M'])
DT = np.array(datos['DT'])
DT = np.array(datos['DT'])
DT = np.array(datos['DT'])
#point = Point(N[0], M[0])
#plt.scatter(N[0],M[0])
#polygon.contains(point)


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


datos['Porosidad'] = puntos
datos.to_csv('eval_petro_output.csv')

#print(datos)
datos.head()
print(datos)
plt.plot(datos['GR'],datos['PROF'], datos['FR'],datos['PROF'])
plt.grid()
plt.show()

np.mean(datos['GR'])


#gr 0 150
#resistivos 0.2 2000
#dt 45 189




import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.01, 5.0, 0.01)
s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = np.sin(4 * np.pi * t)

ax1 = plt.subplot(311)
plt.plot(t, s1)
plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(t, s2)
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

# share x and y
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot(t, s3)
plt.xlim(0.01, 5.0)
plt.show()
