import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#datos
datos = pd.read_csv('eval_petro.csv')
#datos['DT'] = np.around(np.array( 189 - (datos['RHOB'] -1)*datos['M']/0.01 ),   decimals =4)
#datos['N']  = np.around(np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1)),   decimals = 4)
#datos['L']  = np.around(np.array( 0.01 * (189-datos['DT']) / (1-datos['NPHI']) ), decimals =4)

M = np.array( 0.01 * (189-datos['DT'])/(datos['RHOB'] - 1) )
N = np.array( (1 - datos['NPHI']) / (datos['RHOB'] - 1) )
L = np.array( 0.01 * (189 - datos['DT'])/(1 - datos['NPHI']) )
datos['M'] = np.around(M, decimals = 4)
datos['N'] = np.around(N, decimals = 4)
datos['L'] = np.around(L, decimals = 4)

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
az = param_lito(DOLOMIA)[2]
bx = param_lito(CALIZA)[1]
by = param_lito(CALIZA)[0]
bz = param_lito(CALIZA)[2]
cx = param_lito(SILICE)[1]
cy = param_lito(SILICE)[0]
cz = param_lito(SILICE)[2]
dx = param_lito(ARCILLA)[1]
dy = param_lito(ARCILLA)[0]
dz = param_lito(ARCILLA)[2]

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
"""
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
"""

#librerias para trabajar con el poligono
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#vertices del poligono
r1 = DOL_CAL_SIL_FIP = Polygon([(ax, ay), (bx , by), (cx , cy), (ax, ay)])
r2 = DOL_SIL_ARC_FIP = Polygon([(ax, ay), (cx , cy), (dx , dy), (ax, ay)])
r3 = DOL_CAL_FIP_FIS = Polygon([(ax, ay), (bx , by), (bx , 1.2), (ax, 1.2), (ax ,ay)])
r4 = CAL_SIL_FIP_FIS = Polygon([(bx, by), (cx , cy), (cx , 1.2), (bx ,1.2), (bx, by)])


"""definicion de los datos, conversion de datos de serie a numericos tipo array"""
#M = np.array(datos['M'])
#N = np.array(datos['N'])

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
#datos.to_csv('eval_petro_output.csv') #ahhh era para sacar un excel diferente


#manipulacion de datos como arreglos
PROF = np.array(datos['PROF'])
GR   = np.array(datos['GR'])
LLS  = np.array(datos['LLS'])
LLD  = np.array(datos['LLD'])
FR   = np.array(datos['FR'])
DT   = np.array(datos['DT'])
NPHI = np.array(datos['NPHI'])
RHOB = np.array(datos['RHOB'])
#M    = np.array(datos['M'])
#N    = np.array(datos['N'])
#L    = np.array(datos['L'])
Porosidad = np.array(datos['Porosidad'])
#print(datos)
datos.head()


#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv
r1 = str("DOL_CAL_SIL_FIP")
r2 = str("DOL_SIL_ARC_FIP")
r3 = str("DOL_CAL_FIP_FIS")
r4 = str("CAL_SIL_FIP_FIS")




FIP  = np.array([])
VDOL = np.array([])
VCAL = np.array([])
VSIL = np.array([])
VARC = np.array([])
FIS  = np.array([])


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
datos['FIP']  = np.around(FIP, decimals = 4)
datos['FIS']  = np.around(FIS, decimals = 4)

#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv


FIT = FIP + FIS
datos['FIT']  = np.around(FIT, decimals = 4)

Suma = VDOL + VCAL + VSIL + VARC + FIS + FIP
datos['Suma']  = np.around(Suma, decimals = 4)


SumaABS = abs(VDOL) + abs(VCAL) + abs(VSIL) + abs(VARC) + abs(FIS) + abs(FIP)
VDOLR = abs(VDOL)/SumaABS
VCALR = abs(VCAL)/SumaABS
VSILR = abs(VSIL)/SumaABS
VARCR = abs(VARC)/SumaABS
FIPR  = abs(FIP)/SumaABS
FISR  = abs(FIS)/SumaABS
FITR  = abs(FIT)/SumaABS

#
# datos['VDOLR(%)'] = np.around(VDOLR * 100,decimals = 4)
# datos['VCALR(%)'] = np.around(VCALR * 100,decimals = 4)
# datos['VSILR(%)'] = np.around(VSILR * 100,decimals = 4)
# datos['VARCR(%)'] = np.around(VARCR * 100,decimals = 4)
# datos['FIPR(%)']  = np.around(FIPR * 100, decimals = 4)
# datos['FISR(%)']  = np.around(FISR * 100, decimals = 4)
# datos['FITR(%)']  = np.around(FITR * 100, decimals = 4)


datos['VDOLR'] = np.around(VDOLR ,decimals = 4)
datos['VCALR'] = np.around(VCALR ,decimals = 4)
datos['VSILR'] = np.around(VSILR ,decimals = 4)
datos['VARCR'] = np.around(VARCR ,decimals = 4)
datos['FIPR']  = np.around(FIPR  ,decimals = 4)
datos['FISR']  = np.around(FISR  ,decimals = 4)
datos['FITR']  = np.around(FITR  ,decimals = 4)


SumaR = VDOLR + VCALR + VSILR + VARCR + FISR + FIPR
datos['SumaR']  = np.around(SumaR, decimals = 4)

#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv


m   = np.log(1/FR) / np.log(FITR)
FIF = FITR**m
FIENT = FITR - FIF
# datos['m']  = np.around(m, decimals = 4)
# datos['FIF(%)']  = np.around(FIF*100, decimals = 4)
# datos['FIENT(%)']  = np.around(FIENT*100, decimals = 4)

datos['m']  = np.around(m, decimals = 4)
datos['FIF']  = np.around(FIF, decimals = 4)
datos['FIENT']  = np.around(FIENT, decimals = 4)

#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv

FIEFE = FITR*(1-VARCR)
FIEFE_FIF = FIEFE / FIF
FIZ = FITR / (1-FITR)

datos['FIEFE']      = np.around( FIEFE , decimals =4)
datos['FIEFE_FIF']  = np.around( FIEFE_FIF, decimals =4)
datos['FIZ']        = np.around( FIZ, decimals =4)

#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv
#print(datos)
"""
plt.scatter(FITR,FR)
plt.xscale("log")
#plt.yscale("log")
plt.xlim(0.01 ,1)
# plt.ylim(0,1000)
plt.grid()
plt.show()
"""
def beta1(x,y):
    termino1 = x - np.average(x)
    termino2 = y - np.average(y)
    Sxy = sum(termino1*termino2)
    Sxx = sum(termino1*termino1)
    return Sxy/Sxx

def beta0(x,y):
    return np.average(y) - beta1(x,y)*np.average(x)


def plot_recta(x,y):
    b1 = beta1(x,y)
    b0 = beta0(x,y)
    puntos_x = np.linspace(0.01, 1, 100)
    puntos_y = b0+b1*puntos_x
    plt.plot(puntos_x,puntos_y)
    plt.plot(x,y, 'r.')
    # plt.title(-1*np.tan(b1))
    plt.title(np.tan(b1))
    plt.xlabel("Porosidad")
    plt
    plt.ylabel("FR")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.01 ,1)
    plt.ylim(1,1000)
    plt.grid(True,which="both",ls="-")
    plt.show()

"""
plot_recta(FITR,FR)
#plot_recta(FIT,LLD)
#datos['FITR'].describe()
"""

from scipy.stats import linregress

#slope, intercept, r_value, p_value, std_err  = linregress(np.log10(FIT), np.log10(FR))
slope, intercept, r_value, p_value, std_err  = linregress(FITR, FR)
X = np.linspace(0.06,1, 100)
plt.plot(X[:29], intercept + slope*X[:29],'--',  linewidth=1.5, label = 'regresión lineal', color='orange') #'g--',
plt.style.use('ggplot')
plt.plot(FITR, FR, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.ylim(1,1000)
plt.xlim(0.01,1)
plt.xscale("log")
plt.yscale("log")
plt.grid(True,which="both",ls="-", color='0.85')
plt.title('Gráfica 6, FR vs ' + r'$\phi$')
plt.xlabel("Porosidad total")
plt.ylabel("Factor de Resistividad")
plt.plot(X, 19.9543920761623*X**-0.4366084898964984, linewidth=1.5, label='$FR=19.9543\phi^{-0.4366}$', color='c')
plt.legend()
plt.show()


mARCHIE = abs(np.ones(400) * linregress(np.log10(FITR), np.log10(FR))[0])
mARCHIE
ai = FR * FITR ** 0.4366084898964984
a = np.mean(ai)
a
FRcARCHIE = a * FITR **(-1*mARCHIE)
np.mean(FRcARCHIE)
#for style in plt.style.available:
#    print(style)
type(mARCHIE)
Err_mARCHIE = abs((m - abs(mARCHIE))/m) * 100

datos['mARCHIE']    = np.around(mARCHIE, decimals = 4)
datos['FRcARCHIE']  = np.around(FRcARCHIE, decimals = 4)
datos['Err_mARCHIE(%)']= np.around(Err_mARCHIE, decimals = 4)


#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv


mSHELL = 1.87+(0.019/FITR)
FRcSHELL = a * FITR **(-1*mSHELL)
Err_mSHELL = abs((m - abs(mSHELL))/m) * 100

datos['mSHELL']    = np.around(mSHELL, decimals = 4)
datos['FRcSHELL']  = np.around(FRcSHELL, decimals = 4)
datos['Err_mSHELL(%)']= np.around(Err_mSHELL, decimals = 4)




mBORAI = 2.2 - (0.035/(FITR+0.042))
FRcBORAI = a * FITR **(-1*mBORAI)
Err_mBORAI = abs((m - abs(mBORAI))/m) * 100

datos['mBORAI']    = np.around(mBORAI, decimals = 4)
datos['FRcBORAI']  = np.around(FRcBORAI, decimals = 4)
datos['Err_mBORAI(%)']= np.around(Err_mBORAI, decimals = 4)




slope, intercept, r_value, p_value, std_err  = linregress(np.log10(LLD), np.log10(FITR))
mnolog = linregress(LLD, FITR)[0]
intercept_nolog = linregress(LLD, FITR)[1]
mPICKETT = -np.log10(FR)/np.log10(FITR)
mPICKETT = linregress(np.log10(LLD), np.log10(FITR))[0]

#mPICKETT = abs(np.ones(400) * linregress(np.log10(LLD), np.log10(FITR))[0])
#mPICKETT
aPICKETT = FITR * LLD ** mPICKETT
a = np.mean(aPICKETT)
a
X = np.linspace(1,1000, 400)
plt.plot(X, intercept_nolog + mnolog*X,'--',  linewidth=1.5, label = 'regresión lineal', color='orange') #'g--',
plt.style.use('ggplot')
plt.plot(LLD,FITR, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.xlim(0.1,1000)
plt.ylim(0.06,0.11)
plt.xscale("log")
#plt.yscale("log")
plt.grid(True,which="both",ls="-", color='0.85')
plt.title('Gráfica 6, FR vs ' + r'$\phi$')
plt.xlabel("Resistividad DLL")
plt.ylabel("Porosidad total")
plt.plot(X, a*X**mPICKETT, linewidth=1.5, label='$FR=19.9543\phi^{-0.4366}$', color='c')
plt.legend()
plt.show()



mPICKETT = np.mean(-(np.log(FR)/np.log(FITR)))
RW = LLD*FITR**mPICKETT
FRcPICKETT = 1 * FITR **(-1*mPICKETT)
Err_mPICKETT = abs((m - abs(mPICKETT))/m) * 100

datos['mPICKETT']    = np.around(mPICKETT, decimals = 4)
datos['FRcPICKETT']  = np.around(FRcPICKETT, decimals = 4)
datos['Err_mPICKETT(%)']= np.around(Err_mPICKETT, decimals = 4)




Ik = 84105*(FITR**(m+2) /  (1-FITR)**2)


#mraiga, col(47)
mRAIGA = 1.28 + 2 / (np.log10(Ik) + 2)
FRcRAIGA = 1 * FITR **(-1*mRAIGA)
Err_mRAIGA = abs((m - abs(mRAIGA))/m) * 100

datos['mRAIGA']    = np.around(mRAIGA, decimals = 4)
datos['FRcRAIGA']  = np.around(FRcRAIGA, decimals = 4)
datos['Err_mRAIGA(%)']= np.around(Err_mRAIGA, decimals = 4)


TFrFIT = FR * FITR
TFIF_m = FITR * (FITR ** -m)
TFIF_FIENT = FITR / FITR ** m

datos['TFrFIT']     = np.around(TFrFIT, decimals = 4)
datos['TFIF_m']     = np.around(TFIF_m, decimals = 4)
datos['TFIF_FIENT'] = np.around(TFIF_FIENT, decimals = 4)
