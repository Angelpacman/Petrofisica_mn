import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


#captura de los datos
datos = pd.read_csv("eval_petro(0).csv")

#Cálculo de los parametros
M = np.array( 0.01 * (189-datos['DT'])/(datos['RHOB'] - 1) )
N = np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1) )
L = np.array( 0.01 * (189 - datos['DT'])/(1 - datos['NPHI']) )
datos['M'] = np.around(M, decimals = 4)
datos['N'] = np.around(N, decimals = 4)
datos['L'] = np.around(L, decimals = 4)

#Profundidad negativa y armado de referencia para colormap
PROF= np.array(datos['PROF'])  #*-1
col = np.linspace(-1*PROF[0],-1*PROF[-1],400)

#Armado de la figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p3d = ax.scatter(N, M, L, s=30, c=col, marker='.')
colorbar = plt.colorbar(p3d)
colorbar.set_label('metros')
ax.plot(N, L,'r.', zdir='y', zs=0.95, markersize=2)
ax.plot(M, L,'g.', zdir='x', zs=0.66, markersize=2)
ax.plot(N, M,'k.', zdir='z', zs=1.1, markersize=2)
ax.set_xlim([0.50, 0.66])
ax.set_ylim([0.7, 0.95])
ax.set_zlim([1.1, 1.6])
ax.set_xlabel('N')
ax.set_ylabel('M')
ax.set_zlabel('L')
plt.show()
