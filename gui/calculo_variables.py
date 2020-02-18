def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[2] - 1)
    N = (1 - mineral[1]) / (mineral[2] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[1])
    return    np.array([M,N,L])


def MNL(datos):

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

def poligono():
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    #vertices del poligono
    DOL_CAL_SIL_FIP = Polygon([(ax, ay), (bx , by), (cx , cy), (ax, ay)])
    DOL_SIL_ARC_FIP = Polygon([(ax, ay), (cx , cy), (dx , dy), (ax, ay)])
    DOL_CAL_FIP_FIS = Polygon([(ax, ay), (bx , by), (bx , 1.2), (ax, 1.2), (ax ,ay)])
    CAL_SIL_FIP_FIS = Polygon([(bx, by), (cx , cy), (cx , 1.2), (bx ,1.2), (bx, by)])

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
