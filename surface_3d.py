import numpy as np
import matplotlib.pyplot as plt
#Delimitacion de figura con valores mas precisos de minerales

DOLOMIA = np.array([43.5,   0.02,     2.87])
CALIZA  = np.array([47.6,   0.00,     2.71])
SILICE  = np.array([55.5,  -0.035,    2.65])
ARCILLA = np.array([120,    0.33,     2.35])

def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[2] - 1)
    N = (1 - mineral[1]) / (mineral[2] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[1])
    return    np.array([N,M,L])

dol=param_lito(DOLOMIA)
cal=param_lito(CALIZA)
sil=param_lito(SILICE)
arc=param_lito(ARCILLA)


from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # appropriate import to draw 3d polygons
from matplotlib import style

plt.figure('Superficie de valores Ideales',figsize=(5,5))
custom=plt.subplot(111,projection='3d')

# 1. create vertices from points
verts1 = [list((dol, cal, sil))]
verts2 = [list((dol, sil, arc))]
# 2. create 3d polygons and specify parameters
srf1 = Poly3DCollection(verts1, alpha=.25, facecolor='#8e3AAA')
srf2 = Poly3DCollection(verts2, alpha=.25, facecolor='#4c7093')
# 3. add polygon to the figure (current axes)
plt.gca().add_collection3d(srf1)
plt.gca().add_collection3d(srf2)

custom.set_xlim3d(0.4,  0.7)
custom.set_ylim3d(0.4,  0.9)
custom.set_zlim3d(1.0,  1.5)

custom.set_xlabel('N')
custom.set_ylabel('M')
custom.set_zlabel('L')
plt.show()
