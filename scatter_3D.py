import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

datos = pd.read_csv('eval_petro.csv')

datos['DT'] = 189 - (datos['RHOB'] -1)*datos['M']/0.01

datos['N'] = (1 - datos['NPHI'])/(datos['RHOB'] - 1)

datos['L'] = 0.01 * (189 - datos['DT'])/(1-datos['NPHI'])

P_inicial=[0.5051,0.5241,0.5848,0.6273,0.6273,0.5051]
P_final  =[0.702,0.7781,0.8269,0.8091,0.8091,0.702]
P_M1=[0.5241,0.6273]
P_M2=[0.7781,0.8091]
v_x1=[0.5241,0.5241]
v_y1=[0.7781,0.95]
v_x2=[0.5848,0.5848]
v_y2=[0.8269,0.95]
v_x3=[0.6273,0.6273]
v_y3=[0.8091,0.95]



PROF = np.array(datos['PROF'])
GR = np.array(datos['GR'])
LLS = np.array(datos['LLS'])
FR = np.array(datos['FR'])
DT = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
M = np.array(datos['M'])
N = np.array(datos['N'])
L = np.array(datos['L'])


data = (N, M, PROF)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = fig.gca(projection='3d')

for data, color, group in zip(data, colors, groups):
    N, M, PROF = data
    ax.scatter(N, M, PROF, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot 3d scatter plot')
plt.legend(loc=2)
plt.show()
