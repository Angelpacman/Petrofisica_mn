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

DOLOMIA = np.array([43.5,   0.02,   2.87])
CALIZA  = np.array([47.6,   0.00,   2.71])
SILICE  = np.array([55.5,  -0.035,  2.65])
ARCILLA = np.array([120,    0.33,   2.35])

def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[2] - 1)
    N = (1 - mineral[1]) / (mineral[2] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[1])
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

triang_dol_cal_sil_A = [a_x,    b_x,    c_x,    a_x]
triang_dol_cal_sil_B = [a_y,    b_y,    c_y,    a_y]
triang_dol_cal_sil_C = [a_z,    b_z,    c_z,    a_z]


triang_dol_sil_arc_A = [a_x,    c_x,    d_x,    a_x]
triang_dol_sil_arc_B = [a_y,    c_y,    d_y,    a_y]
triang_dol_sil_arc_C = [a_z,    c_z,    d_z,    a_z]

#figura
# plt.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
plt.plot(triang_dol_sil_arc_A, triang_dol_sil_arc_B)
plt.plot(triang_dol_cal_sil_A, triang_dol_cal_sil_B)
plt.plot(v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
plt.plot(datos['N'],datos['M'],marker='o', markersize=2, linestyle='', color='r', label = "M vs N")
#plt.scatter(datos['N'],datos['M'])
#plt.xlim([0.5,0.65])
#plt.ylim([0.7,0.95])
plt.xlim([0.3,1])
plt.ylim([0.4,1])
plt.grid()
plt.xlabel('N')
plt.ylabel('M')
plt.title('M vs N')
plt.show()


#librerias para trabajar con el poligono
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#vertices del poligono
r1 = DOL_CAL_SIL_FIP = Polygon([(a_x, a_y), (b_x , b_y), (c_x , c_y), (a_x, a_y)])
r2 = DOL_SIL_ARC_FIP = Polygon([(a_x, a_y), (c_x , c_y), (d_x , d_y), (a_x, a_y)])
r3 = DOL_CAL_FIP_FIS = Polygon([(a_x, a_y), (b_x , b_y), (b_x , 1.2), (a_x, 1.2), (a_x ,a_y)])
r4 = CAL_SIL_FIP_FIS = Polygon([(b_x, b_y), (c_x , c_y), (c_x , 1.2), (b_x ,1.2), (b_x, b_y)])


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



i = 0
for area in Porosidad:
    if area == r1:
        A = np.array([  [  189,  43.50,  47.50, 55.5  ],
                        [  1.0,   0.02,   0.00, -0.035],
                        [  1.0,   2.87,   2.71,  2.65 ],
                        [  1.0,   1.0 ,   1.0 ,  1.0  ]    ])
        A.shape
        #define matrix B
        b = np.array([  DT[i],
                        NPHI[i],
                        RHOB[i],
                        1       ])
        X    = np.linalg.solve(A, b)
        FIP  = np.append(FIP,   [X[0]], axis=0)
        VDOL = np.append(VDOL,  [X[1]], axis=0)
        VSIL = np.append(VSIL,  [X[3]], axis=0)
        VARC = np.append(VARC,  [0],    axis=0)
        VCAL = np.append(VCAL,  [X[2]], axis=0)
        FIS  = np.append(FIS,   [0],    axis=0)



    else:
        if area == r2:
            A = np.array([  [  189, 43.50,  55.5,   120   ],
                            [  1.0,  0.02,  -0.035, 0.33  ],
                            [  1.0,  2.87,   2.65,  2.35  ],
                            [  1.0,  1.0,    1.0,   1.0   ]   ])
            A.shape
            #define matrix B
            b = np.array([  DT[i],
                            NPHI[i],
                            RHOB[i],
                            1       ])
            X    = np.linalg.solve(A, b)
            FIP  = np.append(FIP,   [X[0]], axis=0)
            VDOL = np.append(VDOL,  [X[1]], axis=0)
            VSIL = np.append(VSIL,  [X[2]], axis=0)
            VARC = np.append(VARC,  [X[3]], axis=0)
            VCAL = np.append(VCAL,  [0],    axis=0)
            FIS  = np.append(FIS,   [0],    axis=0)



        else:
            if area == r3:
                A = np.matrix([ [189,   45.55,  43.5,   47.6    ],
                                [1.0,    1.0,    0.02,   0.0    ],
                                [1.0,    1.0,    2.87,   2.71   ],
                                [1.0,    1.0,    1.0,    1.0    ]  ])
                A.shape
                #define matrix B
                b = np.array([  DT[i],
                                NPHI[i],
                                RHOB[i],
                                1       ])
                X    = np.linalg.solve(A, b)
                FIP  = np.append(FIP,   [X[0]], axis=0)
                VDOL = np.append(VDOL,  [X[2]], axis=0)
                VSIL = np.append(VSIL,  [0],    axis=0)
                VARC = np.append(VARC,  [0],    axis=0)
                VCAL = np.append(VCAL,  [X[3]], axis=0)
                FIS  = np.append(FIS,   [X[1]], axis=0)



            else:
                if area == r4:
                    A = np.array([  [189,  51.55,  47.6,   55.5     ],
                                    [1.0,   1.0,    0.0,   -0.035   ],
                                    [1.0,   1.0,    2.71,   2.65    ],
                                    [1.0,   1.0,    1.0,    1.0     ]  ])
                    A.shape
                    #define matrix B
                    b = np.array([  DT[i],
                                    NPHI[i],
                                    RHOB[i],
                                    1       ])
                    X    = np.linalg.solve(A, b)
                    FIP  = np.append(FIP,   [X[0]], axis=0)
                    VDOL = np.append(VDOL,  [0],    axis=0)
                    VSIL = np.append(VSIL,  [X[3]], axis=0)
                    VARC = np.append(VARC,  [0],    axis=0)
                    VCAL = np.append(VCAL,  [X[2]], axis=0)
                    FIS  = np.append(FIS,   [X[1]], axis=0)


                else:
                    FIP  = np.append(FIP,   [np.NaN], axis=0)
                    VDOL = np.append(VDOL,  [np.NaN], axis=0)
                    VSIL = np.append(VSIL,  [np.NaN], axis=0)
                    VARC = np.append(VARC,  [np.NaN], axis=0)
                    VCAL = np.append(VCAL,  [np.NaN], axis=0)
                    FIS  = np.append(FIS,   [np.NaN], axis=0)
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

datos['m']      = np.around(m, decimals = 4)
datos['FIF']    = np.around(FIF, decimals = 4)
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
X = np.linspace(0.01,1, 100)
plt.plot(X[:29], intercept + slope*X[:29],'--', linewidth=1.5, label = 'regresión lineal', color='orange') #'g--',
plt.style.use('ggplot')
plt.plot(FITR, FR, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.ylim(1,1000)
plt.xlim(0.01,1)
plt.xscale("log")
plt.yscale("log")
plt.grid(True,which="both",ls="-", color='0.85')
plt.title('Gráfico de Archie, FR vs ' + r'$\phi$')
plt.xlabel("Porosidad total")
plt.ylabel("Factor de Resistividad")
plt.plot(X, 12.930633472943219*X**-0.42514508, linewidth=1.5, label='$FR=12.9306\phi^{-0.4251}$', color='c')#0.4366084898964984 19.9543920761623
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
mPICKETT
X = np.linspace(1,1000, 400)
plt.plot(X, intercept_nolog + mnolog*X,'--',  linewidth=1.5, label = 'regresión lineal', color='orange') #'g--',
plt.style.use('ggplot')
plt.plot(LLD,FITR, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.xlim(1,1000)
plt.ylim(0.001,1)
plt.xscale("log")
plt.yscale("log")
plt.grid(True,which="both",ls="-", color='0.85')
plt.title('Gráfico de Pickett, DLL vs ' + r'$\phi$')
plt.xlabel("Resistividad DLL")
plt.ylabel("Porosidad total")
#plt.plot(X, a*X**mPICKETT, linewidth=1.5, color='c')#, label='$FR=19.9543\phi^{-0.4366}$'
#plt.plot(X, 1.3*X**-1.9, linewidth=1.5, color='c')#, label='$FR=19.9543\phi^{-0.4366}$'
#plt.plot(X, 3*X**-2.5, linewidth=1.5, color='c')#, label='$FR=19.9543\phi^{-0.4366}$'
plt.plot(X, 12.8*X**-3.1, linewidth=1.5, color='c', label='si $\phi = 1; Rw = 2.1708$')#, label='$FR=19.9543\phi^{-0.4366}$'

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

#col(53)
# CoParPIRSON = (FITR - FIPR) / (FITR * (1-FIPR))
FImatTAREK  = (FITR**m - FITR) / (FITR**m - 1)
FIfracTAREK = (FITR**(m+1) - FITR**m) / (FITR**m - 1)
CoParPIRSON = ((FImatTAREK + FIfracTAREK) - FImatTAREK) / ((FImatTAREK + FIfracTAREK) * (1-FImatTAREK))
CoParCLASE  = FImatTAREK / (FImatTAREK + FIfracTAREK)
CoParTAREK  =( FIfracTAREK / (FImatTAREK + FIfracTAREK))
Err_CoParLITO = np.ones(400)
i = 0
for coeficiente in CoParPIRSON:
    if CoParPIRSON[i] == 0.0:
        # Err_CoParLITO[i] = abs((CoParPIRSON[i] - CoParTAREK[i]) / CoParPIRSON[i]) * 100
        #np.append(Err_CoParLITO, np.NaN)
        Err_CoParLITO[i] = np.NaN
    else:
        #np.append(Err_CoParLITO, abs((CoParTAREK[i] - CoParPIRSON[i]) / CoParTAREK[i])*100)
        Err_CoParLITO[i] = abs((CoParPIRSON[i] - CoParTAREK[i]) / CoParPIRSON[i])*100
    i += 1
#Err_CoParLITO = abs((CoParPIRSON - CoParTAREK) / CoParPIRSON) * 100
#Err_CoParLITO
InvTORTUOSIDAD = TFrFIT ** -1
CONECTIVIDAD = np.log10(1 - FR ** -1) / np.log10(1-FITR)


datos['CoParPIRSON']    = np.around(CoParPIRSON,decimals = 4)
datos['CoParCLASE']     = np.around(CoParCLASE, decimals = 4)
datos['FImatTAREK']     = np.around(FImatTAREK, decimals = 4)
datos['FIfracTAREK']    = np.around(FIfracTAREK,decimals = 4)
datos['CoParTAREK']     = np.around(CoParTAREK, decimals = 4)
datos['Err_CoParLITO']  = np.around(Err_CoParLITO,decimals = 4)
datos['InvTORTUOSIDAD'] = np.around(InvTORTUOSIDAD,decimals = 4)
datos['CONECTIVIDAD']   = np.around(CONECTIVIDAD,   decimals = 4)

#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv



datos['Porosidad'].value_counts()
datos['FITR'].value_counts()




#col(61) indice de intensidad de fracturamiento
IIF = CoParPIRSON*FITR
w = (FITR**m - FITR**(m-1)) / (FITR**m - 1) #indice de almacenamiento de fractura
C = ((1-FITR)**2) / (FR*FITR - 1) #estructura porosa eficiente
FIELE = C*FITR / ((1-FITR)**2 + C)
#Ik
CteKC = TFrFIT * m
datos['IIF']    = np.around(IIF,    decimals = 4)
datos['w']      = np.around(w,      decimals = 4)
datos['C']      = np.around(C,      decimals = 4)
datos['FIELE']  = np.around(FIELE,  decimals = 4)
datos['Ik']     = np.around(Ik,     decimals = 4)
datos['CteKC']  = np.around(CteKC,  decimals = 4)



#col(67)
RGP35 = 2.66*(Ik/(FITR*100))**0.45
TGP = []
i = 0
for numero in RGP35:
    if numero < 0.5:
        TGP.append("Nano")
    else:
        if numero > 0.5 and numero < 2:
            TGP.append("Micro")
        else:
            if numero > 2 and numero < 4:
                TGP.append("Meso")
            else:
                if numero > 4 and numero < 10:
                    TGP.append("Macro")
                else:
                    TGP.append("Mega")
#TGP
SP = ((4.46*10**10)/((FR**2.2)*(FITR**1.2) * Ik))**(1/2)

GS = (6*(1-FITR))/SP

ICY = 0.0314*(Ik/FITR)**(1/2)
IZF = ICY / FIZ
IRE = (FITR/FR)**(1/2)
IZE = IRE/FIZ
m_Evans = (0.0811 * FITR) + 1.4328


datos['RGP35']  = np.around(RGP35,  decimals = 4)
datos['TGP']    = TGP
datos['SP']     = np.around(SP,     decimals = 4)
datos['GS']     = np.around(GS,     decimals = 4)
datos['ICY']    = np.around(ICY,    decimals = 4)
datos['IZF']    = np.around(IZF,    decimals = 4)
datos['IRE']    = np.around(IRE,    decimals = 4)
datos['IZE']    = np.around(IZE,    decimals = 4)
datos['m_Evans']= np.around(m_Evans,decimals = 4)


#datos.to_csv('eval_petro_output.csv') #exportando al archivo csv

datos.info()

plt.title("Gráfico de frecuencia de M")
plt.xlabel("M")
plt.xlim(0.5, 0.9)
plt.ylabel("Frecuencia")
plt.hist(M, bins = 30 )
plt.show()
plt.title("Gráfico de frecuencia de N")
plt.xlabel("N")
plt.xlim(0.5, 0.9)
plt.ylabel("Frecuencia")
plt.hist(N, bins = 30 )
plt.show()
plt.title("Gráfico de frecuencia de L")
plt.xlabel("L")
plt.xlim(1, 1.5)
plt.ylabel("Frecuencia")
plt.hist(L, bins = 30 )
plt.show()


plt.title("m vs FIENT")
plt.xlabel("FIENT")
plt.ylabel("m")
plt.plot(FIENT, m, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()



plt.title("$\phi f$ vs T")
plt.xlabel("FIF")
plt.ylabel("T")
plt.plot(FIF, TFrFIT, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()



plt.title("T vs m")
plt.xlabel("T")
plt.ylabel("m")
plt.plot(TFrFIT,m, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


plt.title("1/T vs Coeficente de partición")
plt.xlabel("1/T")
plt.ylabel("Coeficiente de partición")
plt.plot(InvTORTUOSIDAD,CoParTAREK, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


plt.title("CoParTAREK vs Conectividad")
plt.xlabel("Conectividad 'r'")
plt.ylabel("CoParTAREK")
plt.plot(CONECTIVIDAD,CoParTAREK, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


plt.title("T vs CoParTAREK")
plt.xlabel("CoParTAREK")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("T")
plt.grid(True, which ='both')
plt.plot(CoParTAREK,TFrFIT, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


P = (FITR ** 3)/((1-FITR)**2)

plt.title("Ik vs P")
plt.xlabel("P")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Indice de permeabilidad Ik")
plt.grid(True, which ='both')
plt.plot(P,Ik, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()





plt.title("Ik vs $\phi$ent")
plt.xlabel("$\phi$ent")
plt.xscale("log")
plt.xlim(0.0001,0.1)
plt.yscale("log")
plt.ylabel("Ik")
plt.grid(True, which ='both')
plt.plot(FIENT,Ik, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()



plt.title("SP vs CteKC")
plt.xlabel("Constante K-C")
plt.xscale("log")
plt.xlim(0.01,100)
plt.yscale("log")
plt.ylabel("SP")
plt.grid(True, which ='both')
plt.plot(CteKC,SP, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()

plt.title("Ik vs FR")
plt.xlabel("Factor de resistividad FR")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Indice de permeabilidad Ik")
plt.xlim(1,1000)
plt.grid(True, which ='both')
plt.plot(FR,Ik, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


plt.title("SP vs GS")
plt.xlabel("Tamaño de grano GS")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Superficie especifica poral")
#plt.xlim(1,1000)
plt.grid(True, which ='both')
plt.plot(GS,SP, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


plt.title("GS vs mIFV")
plt.xlabel("mIFV")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Tamaño de grano GS")
plt.ylim(0.00001,0.01)
plt.xlim(0.1,10)
plt.grid(True, which ='both')
plt.plot(m,GS, marker = 'o', markersize = '1.5', linestyle='', color = 'r')
plt.show()


x = LLD
y = LLD/LLS

plt.scatter(x,y)
plt.plot((300,450),(2.5,1.75))
plt.plot((0,450),(1,1))
plt.plot((0,200),(2,1))
plt.plot((300,450),(2.5,1.75))

plt.grid(True, which = 'both')
plt.xlabel('LLD')
plt.ylabel('LLD / LLS')
plt.title('Resistividad vs relación Laterlog')
plt.xlim(0,450)
plt.ylim(0,2.5)
plt.show()

plt.plot(LLD,PROF*-1,LLS,PROF*-1)
plt.show()


plt.plot(y,PROF)
plt.plot((0,0),(4300,4250))
plt.xlabel('Relacion LLD / LLS')
plt.ylabel('Profundidad')
plt.xlim(-1.5,3.5)
plt.show()

#FImatRasmus= 100.0 - (np.sqrt(4.0*((100.0)*(1-0.02))))/(2*(100.0*(1-0.02)))
#FImatRasmus


FRcEvans = 1 * FITR **(-1*m_Evans)
Err_m_Evans = abs((m - abs(m_Evans))/m) * 100
datos['FRcEvans']  = np.around(FRcEvans, decimals = 4)
datos['Err_m_Evans(%)']= np.around(Err_m_Evans, decimals = 4)


_Por_ = FITR / ((1-FITR)**2)   #78
m_Pivote = 2.0 + 12.5*(1 - (FIPR/FITR))*(FITR - 0.07)
FRcPivote = 1 * FITR **(-1*m_Pivote)
Err_m_Pivote = abs((m - abs(m_Pivote))/m) * 100

datos['Por*']           = np.around(_Por_,      decimals = 4)
datos['m_Pivote']       = np.around(m_Pivote,   decimals = 4)
datos['FRcPivote']      = np.around(FRcPivote,  decimals = 4)
datos['Err_m_Pivote(%)']= np.around(Err_m_Pivote, decimals = 4)


Laterlog = LLD/LLS

FiFracCPR = 1- ((FR - 1)/FR)**(1/CONECTIVIDAD)
FImatCPR = FITR - FiFracCPR
P = LLD*FITR**mPICKETT
P_1medio = (LLD * FITR**mPICKETT)**0.5
Pc = 255.86*(Ik/FITR)


datos['LLD/LLS']    = np.around(Laterlog,   decimals = 4)
datos['FiFracCPR']  = np.around(FiFracCPR,  decimals = 4)
datos['FImatCPR']   = np.around(FImatCPR,   decimals = 4)
datos['P']          = np.around(P,          decimals = 4)
datos['P_1medio']   = np.around(P_1medio,   decimals = 4)
datos['Pc']         = np.around(Pc,         decimals = 4)


Dm = 6*(1-FITR)/SP
FIF_FIELE = FIF / FIELE
FIEFE_FITR = FIEFE / FITR
G = 0.887
n = 1.87
SwArchie_modif = (G*0.20234/FITR)**(-1/n)
S_HCS = 1-SwArchie_modif
FIT_Sw = FIT * SwArchie_modif
C_ = np.mean(FIT_Sw)
SwIRE = C_ / FITR
TamBLO = (1-FIF)**(1/3)
Pc_ = 151.35*(Ik/(FITR*100))**-0.407
A_ = 2.03
B_ = 0.9
log_a = (A_*np.log10(FITR) + np.log10(FR)) / (1+B_*np.log10(FITR))
mOGR = 1.13*log_a


datos['Dm']             = np.around(Dm,         decimals = 4)
datos['FIF_FIELE']      = np.around(FIF_FIELE,  decimals = 4)
datos['FIEFE_FITR']     = np.around(FIEFE_FITR, decimals = 4)
datos['SwArchie_modif'] = np.around(SwArchie_modif,decimals = 4)
datos['S_HCS']          = np.around(S_HCS,      decimals = 4)
datos['FIT_Sw']         = np.around(FIT_Sw,     decimals = 4)
datos['SwIRE']          = np.around(SwIRE,      decimals = 4)
datos['TamBLO']         = np.around(TamBLO,     decimals = 4)
datos['Pc_']            = np.around(Pc_,        decimals = 4)
datos['log_a']          = np.around(log_a,      decimals = 4)
datos['mOGR']           = np.around(mOGR,       decimals = 4)


datos.to_csv('eval_petro_output.csv') #exportando al archivo csv
