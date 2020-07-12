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

DOLOMIA = np.array([43.5,   2.87,   0.02])
CALIZA  = np.array([47.6,   2.71,   0.00])
SILICE  = np.array([55.5,   2.65,  -0.035])
ARCILLA = np.array([120,    2.35,   0.33])

def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[1] - 1)
    N = (1 - mineral[2]) / (mineral[1] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[2])
    return    np.array([N,M,L])

a_x = param_lito(DOLOMIA)[0]
a_y = param_lito(DOLOMIA)[1]
a_z = param_lito(DOLOMIA)[2]

b_x = param_lito(CALIZA)[0]
b_y = param_lito(CALIZA)[1]
b_z = param_lito(CALIZA)[2]

c_x = param_lito(SILICE)[0]
c_y = param_lito(SILICE)[1]
c_z = param_lito(SILICE)[2]

d_x = param_lito(ARCILLA)[0]
d_y = param_lito(ARCILLA)[1]
d_z = param_lito(ARCILLA)[2]

P_inicial=[a_x, b_x,    c_x,    d_x,    a_x]
P_final  =[a_y, b_y,    c_y,    d_y,    a_y]
P_M1=[a_x,c_x]
P_M2=[a_y,c_y]
v_x1=[a_x,a_x]
v_y1=[a_y,1]
v_x2=[b_x,b_x]
v_y2=[b_y,1]
v_x3=[c_x,c_x]
v_y3=[c_y,1]

tirang_dol_cal_sil_A = [a_x,    b_x,    c_x,    a_x]
tirang_dol_cal_sil_B = [a_y,    b_y,    c_y,    a_y]
tirang_dol_cal_sil_C = [a_z,    b_z,    c_z,    a_z]


triang_dol_sil_arc_A = [a_x,    c_x,    d_x,    a_x]
triang_dol_sil_arc_B = [a_y,    c_y,    d_y,    a_y]
triang_dol_sil_arc_C = [a_z,    c_z,    d_z,    a_z]


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
c_bar = plt.colorbar(p2d)
c_bar.set_label('metros')


#aqui va la figura pero en proyeccion 3D de M vs N
fig = plt.figure()
ay = fig.add_subplot(111, projection='3d')
p3d = ay.scatter(N, M, -1*PROF, s=40, c=col, marker='.')
#ay.invert_zaxis()
ay.set_xlabel('N')
ay.set_ylabel('M')
ay.set_zlabel('Profundidad')
colorbar = plt.colorbar(p3d)
colorbar.set_label('metros')


fig = plt.figure()
az  = fig.add_subplot(111, projection='3d')
colL= np.linspace(L[0],L[-1],400)
p3d = az.scatter(N, M, L, s=40, c=col, marker='.')
#ay.invert_zaxis()
"""Este bloque agregado a la grafica 3D MNL dibuja la superficie de los vertices
dol, cal, sil, arc."""
dol = param_lito(DOLOMIA)
cal = param_lito(CALIZA)
sil = param_lito(SILICE)
arc = param_lito(ARCILLA)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # appropriate import to draw 3d polygons
from matplotlib import style
# 1. Add vertix
verts1 = [list((dol, cal, sil))]
verts2 = [list((dol, sil, arc))]
# 2. create 3d polygons and specify parameters
srf1 = Poly3DCollection(verts1, alpha=.25, facecolor='#ff5233')
srf2 = Poly3DCollection(verts2, alpha=.25, facecolor='#4c7093')
# 3. add polygon to the figure (current axes)
plt.gca().add_collection3d(srf1)
plt.gca().add_collection3d(srf2)
#Proyeccion de lineas de la superficie dol cal sil arc
az.plot(triang_dol_sil_arc_A,triang_dol_sil_arc_B, zs=min(L), zdir='z', label='dol-sil-arc')
az.plot(tirang_dol_cal_sil_A,tirang_dol_cal_sil_B, zs=min(L), zdir='z', label='dol-cal-sil')
#az.plot(tirang_dol_cal_sil_B,tirang_dol_cal_sil_C, zs=min(N), zdir='x',)
az.legend()
az.set_zlim(min(L), max(L))
""""""
# az.set_xlim(min(N), max(N))
# az.set_ylim(min(M), max(M))
# az.set_zlim(min(L), max(L))

az.set_xlabel('N')
az.set_ylabel('M')
az.set_zlabel('L')
cb = plt.colorbar(p3d)
cb.set_label('metros')
plt.show()



##ML
P_M  =  [0.7781,    0.8269, 0.8091,    0.7781]
P_L  =  [1.4847,    1.414,  1.2898,    1.4847]
P__M =  [a_y,    c_y,     d_y,     a_y]
P__L =  [a_z,    c_z,     d_z,     a_z]

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.title("Grafico M vs L")
ax.plot(P__M, P__L)
ax.plot(P_M,P_L)
ax.grid()
ax.set_xlabel('M')
ax.set_ylabel('L')
ax.scatter(M, L, s=10, c = col, marker='o')
#p2d = ax.scatter(M, L, s=10, c = col, marker='o')
color_b = plt.colorbar(p2d)
color_b.set_label('metros')

##NL
P_N  = [0.5241,0.5848, 0.6273, 0.5241]
P_L  = [1.4847, 1.414, 1.2898, 1.4847]
P__N = [a_x,    c_x,     d_x,     a_x]
P__L = [a_z,    c_z,     d_z,     a_z]

coli = np.linspace(1,99,400)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.title("Grafico N vs L")
ax.plot(P__N,P__L)
ax.plot(P_N,P_L)
ax.grid()
ax.set_xlabel('N')
ax.set_ylabel('L')
ax.scatter(N, L, s=10, c = col, marker='o')
#p2d = ax.scatter(M, L, s=10, c = coli, marker='o')
color_b = plt.colorbar(p2d)
color_b.set_label('metros')
plt.show()


"""from mayavi import mlab
#mlab.points3d(N, M, L, scale_factor = 0.01)
#mlab.gcf()

s = mlab.points3d(N, M, L, mode = 'point', extent = [0,1,0,1,0,1])
mlab.axes(s, ranges = [min(N), max(N), min(M), max(M), min(L), max(L)])

mlab.show()
"""
