import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#datos
datos = pd.read_csv('eval_petro(0).csv')
datos.head()
# datos['DT'] = np.around(np.array( 189 - (datos['RHOB'] -1)*datos['M']/0.01 ),   decimals =4)
# datos['N']  = np.around(np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1)),   decimals = 4)
# datos['L']  = np.around(np.array( 0.01 * (189-datos['DT']) / (1-datos['NPHI']) ), decimals =4)
datos['M'] = np.array( 0.01 * (189-datos['DT'])/(datos['RHOB'] - 1) )
datos['N'] = np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1) )
datos['L'] = np.array( 0.01 * (189 - datos['DT'])/(1 - datos['NPHI']) )

#manipulacion de datos como arreglos
PROF = np.array(datos['PROF'])
GR   = np.array(datos['GR'])
LLS  = np.array(datos['LLS'])
LLD  = np.array(datos['LLD'])
FR   = np.array(datos['FR'])
DT   = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
RHOB = np.array(datos['RHOB'])
M    = np.array(datos['M'])
N    = np.array(datos['N'])
L    = np.array(datos['L'])


datos.head()
plt.plot(GR,PROF,FR,PROF)
plt.grid()
plt.show()
np.mean(GR)


#gr 0 150
#resistivos 0.2 2000
#dt 45 189

#prueba de ploteo con registros xdxdxdddd
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
