import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


#captura de los datos
datos = pd.read_csv("eval_petro.csv")
# datos = pd.read_csv("gabs.csv")
#datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01
#datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)
#datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])

M = np.array( 0.01 * (189-datos['DT'])/(datos['RHOB'] - 1) )
N = np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1) )
L = np.array( 0.01 * (189 - datos['DT'])/(1 - datos['NPHI']) )
datos['M'] = np.around(M, decimals = 4)
datos['N'] = np.around(N, decimals = 4)
datos['L'] = np.around(L, decimals = 4)

#definicion de los puntos que van a ser los contenedores del poligono

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
v_y1=[ay,1]
v_x2=[bx,bx]
v_y2=[by,1]
v_x3=[cx,cx]
v_y3=[cy,1]



PROF= np.array(datos['PROF'])  #*-1
#z = -1*PROF
col = np.linspace(-1*PROF[0],-1*PROF[-1],400)
#colP = np.linspace(PROF[-1],PROF[0],400)
#colL = np.linspace(L[-1],L[0],400)

#aqui va figura en 2D
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
ax.grid()
ax.set_xlabel('N')
ax.set_ylabel('M')
ax.scatter(N, M, s=10, c = col, marker='o')
p2d = ax.scatter(N, M, s=10, c = col, marker='o')
plt.colorbar(p2d)


#aqui va la figura pero en proyeccion 3D de M vs N
fig = plt.figure()
ay = fig.add_subplot(111, projection='3d')
p3d = ay.scatter(N, M, -1*PROF, s=40, c=col, marker='.')
#ay.invert_zaxis()
ay.set_xlabel('N')
ay.set_ylabel('M')
ay.set_zlabel('Profundidad')
#plt.colorbar(p3d)
##plt.show()


fig = plt.figure()
az = fig.add_subplot(111, projection='3d')
colL = np.linspace(L[0],L[-1],400)
p3d = az.scatter(N, M, L, s=40, c=col, marker='.')
#ay.invert_zaxis()
az.set_xlabel('N')
az.set_ylabel('M')
az.set_zlabel('L')
#plt.colorbar(p3d)
plt.show()



##ML
P_M  =[0.7781,0.8269,0.8091,0.7781]
P_L = [1.4847, 1.414, 1.2898, 1.4847]
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.title("Grafico M vs L")
ax.plot(P_M,P_L)
ax.grid()
ax.set_xlabel('M')
ax.set_ylabel('L')
ax.scatter(M, L, s=10, c = col, marker='o')
#p2d = ax.scatter(M, L, s=10, c = col, marker='o')
plt.colorbar(p2d)

##NL
P_N = [0.5241,0.5848, 0.6273, 0.5241]
P_L = [1.4847, 1.414, 1.2898, 1.4847]
coli = np.linspace(1,99,400)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.title("Grafico N vs L")
ax.plot(P_N,P_L)
ax.grid()
ax.set_xlabel('N')
ax.set_ylabel('L')
ax.scatter(N, L, s=10, c = col, marker='o')
#p2d = ax.scatter(M, L, s=10, c = coli, marker='o')
plt.colorbar(p2d)
plt.show()


"""from mayavi import mlab
#mlab.points3d(N, M, L, scale_factor = 0.01)
#mlab.gcf()

s = mlab.points3d(N, M, L, mode = 'point', extent = [0,1,0,1,0,1])
mlab.axes(s, ranges = [min(N), max(N), min(M), max(M), min(L), max(L)])

mlab.show()
"""
