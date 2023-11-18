import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PyQt5 import uic, QtWidgets

qtCreatorFile = "petrofisica.ui" # Nombre del archivo aquí.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

#Definiciones Fuera de la clase porque al paracer se necesitan de manera global
def param_lito(mineral):
    M = 0.01 * (189-mineral[0])/(mineral[1] - 1)
    N = (1 - mineral[2]) / (mineral[1] - 1)
    L = 0.01 * (189 - mineral[0])/(1 - mineral[2])
    return    np.array([N,M,L])

DOLOMIA = np.array([43.5,   2.87,   0.02])
CALIZA  = np.array([47.6,   2.71,   0.00])
SILICE  = np.array([55.5,   2.65,  -0.035])
ARCILLA = np.array([120,    2.35,   0.33])

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


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        #vamos a poner los botones que nos hacen falta
        self.boton1.clicked.connect(self.getCSV)
        self.boton2.clicked.connect(self.getXLSX)
        self.boton3.clicked.connect(self.PlotMN)
        self.boton4.clicked.connect(self.PlotMNL3D)
        self.boton5.clicked.connect(self.PlotSuperficies)


    #definimos la funcion para obterner el archivo getCSV
    def getCSV(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/angelr/Documentos/documentos/evaluacion_petrofisica/')
        if filePath != "":
            print ("Dirección",filePath) #Opcional imprimir la dirección del archivo
            self.datos = pd.read_csv(str(filePath))


    #definimos la funcion para obterner el archivo getXLSX
    def getXLSX(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/angelr/Documentos/documentos/evaluacion_petrofisica/')
        if filePath != "":
            print ("Dirección",filePath) #Opcional imprimir la dirección del archivo
            self.datos = pd.read_excel(str(filePath))

    def PlotMN(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LinearSegmentedColormap
        M = np.array( 0.01 * (189-self.datos['DT'])/(self.datos['RHOB'] - 1) )
        N = np.array( (1 - self.datos['NPHI']) / (self.datos['RHOB'] - 1) )
        L = np.array( 0.01 * (189 - self.datos['DT'])/(1 - self.datos['NPHI']) )
        self.datos['M'] = np.around(M, decimals = 4)
        self.datos['N'] = np.around(N, decimals = 4)
        self.datos['L'] = np.around(L, decimals = 4)



        PROF= np.array(self.datos['PROF'])  #*-1
        col = np.linspace(-1*PROF[0],-1*PROF[-1],len(PROF))

        #aqui va figura en 2D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
        ax.plot(triang_dol_sil_arc_A, triang_dol_sil_arc_B)
        ax.plot(triang_dol_cal_sil_A, triang_dol_cal_sil_B)
        ax.plot(v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
        ax.grid()
        ax.set_xlabel('N')
        ax.set_ylabel('M')
        ax.scatter(N, M, s=10, c = col, marker='o')
        p2d = ax.scatter(N, M, s=10, c = col, marker='o')
        cb = plt.colorbar(p2d)
        cb.set_label('Profundidad de los datos en el registro')
        ax.set_title('Diagrama M vs N')
        plt.show()



    def PlotMNL3D(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LinearSegmentedColormap
        M = np.array( 0.01 * (189-self.datos['DT'])/(self.datos['RHOB'] - 1) )
        N = np.array( (1 - self.datos['NPHI']) / (self.datos['RHOB'] - 1) )
        L = np.array( 0.01 * (189 - self.datos['DT'])/(1 - self.datos['NPHI']) )
        self.datos['M'] = np.around(M, decimals = 4)
        self.datos['N'] = np.around(N, decimals = 4)
        self.datos['L'] = np.around(L, decimals = 4)

        PROF= np.array(self.datos['PROF'])  #*-1
        col = np.linspace(-1*PROF[0],-1*PROF[-1],400)

        #aqui va la figura pero en proyeccion 3D de M vs N
        fig = plt.figure()
        ay = fig.add_subplot(111, projection='3d')
        p3d = ay.scatter(N, M, -1*PROF, s=40, c=col, marker='.')
        ay.plot(triang_dol_sil_arc_A,triang_dol_sil_arc_B, zs=-1*max(PROF), zdir='z', label='dol-sil-arc')
        ay.plot(triang_dol_cal_sil_A,triang_dol_cal_sil_B, zs=-1*max(PROF), zdir='z', label='dol-cal-sil')
        ay.legend()
        ay.set_zlim(min(-1*PROF), max(-1*PROF))
        ay.set_xlabel('N')
        ay.set_ylabel('M')
        ay.set_zlabel('Profundidad')
        cb = plt.colorbar(p3d)
        cb.set_label('Profundidad de los datos en el registro')
        ay.set_title('Plano M,N vs Profundidad')
        ##plt.show()


        fig = plt.figure()
        az = fig.add_subplot(111, projection='3d')
        colL = np.linspace(L[0],L[-1],400)
        p3d = az.scatter(N, M, L, s=40, c=col, marker='.')
        """Este bloque agregado a la grafica 3D MNL dibuja la superficie de los vertices
        dol, cal, sil, arc."""
        dol=param_lito(DOLOMIA)
        cal=param_lito(CALIZA)
        sil=param_lito(SILICE)
        arc=param_lito(ARCILLA)

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
        az.plot(triang_dol_sil_arc_A,triang_dol_sil_arc_B, zs=min(L), zdir='z', label='dol-sil-arc')
        az.plot(triang_dol_cal_sil_A,triang_dol_cal_sil_B, zs=min(L), zdir='z', label='dol-cal-sil')
        #az.plot(triang_dol_cal_sil_B,triang_dol_cal_sil_C, zs=min(N), zdir='x',)
        az.legend()
        az.set_zlim(min(L), max(L))
        """"""
        az.set_xlabel('N')
        az.set_ylabel('M')
        az.set_zlabel('L')
        cb = plt.colorbar(p3d)
        cb.set_label('Profundidad de los datos en el registro')
        az.set_title('Cubo M,N,L')
        plt.show()


    def PlotSuperficies(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LinearSegmentedColormap
        M = np.array( 0.01 * (189-self.datos['DT'])/(self.datos['RHOB'] - 1) )
        N = np.array( (1 - self.datos['NPHI']) / (self.datos['RHOB'] - 1) )
        L = np.array( 0.01 * (189 - self.datos['DT'])/(1 - self.datos['NPHI']) )
        self.datos['M'] = np.around(M, decimals = 4)
        self.datos['N'] = np.around(N, decimals = 4)
        self.datos['L'] = np.around(L, decimals = 4)




        PROF= np.array(self.datos['PROF'])  #*-1
        col = np.linspace(-1*PROF[0],-1*PROF[-1],400)

        #aqui va figura en 2D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
        ax.plot(triang_dol_sil_arc_A, triang_dol_sil_arc_B)
        ax.plot(triang_dol_cal_sil_A, triang_dol_cal_sil_B)
        #ax.plot(v_x1,v_y1,v_x2,v_y2,v_x3,v_y3) lineas hacia arriba que no sirven :(
        ax.grid()
        ax.set_xlabel('N')
        ax.set_ylabel('M')
        ax.scatter(N, M, s=10, c = col, marker='o')
        p2d = ax.scatter(N, M, s=10, c = col, marker='o')
        color_b = plt.colorbar(p2d)
        color_b.set_label('Profundidad (metros)')
        ax.set_title('Diagrama M vs N')
        plt.show()


        ##ML
        P_M  =  [0.7781,    0.8269, 0.8091,    0.7781]
        P_L  =  [1.4847,    1.414,  1.2898,    1.4847]
        P__M =  [a_y,    c_y,     d_y,     a_y]
        P__L =  [a_z,    c_z,     d_z,     a_z]
        fig = plt.figure()
        ay = fig.add_subplot(111)
        ay.plot(P__M,P__L)
        ay.plot(P_M,P_L)
        ay.grid()
        ay.set_xlabel('M')
        ay.set_ylabel('L')
        ay.scatter(M, L, s=10, c = col, marker='o')
        p2d_ML = ay.scatter(M, L, s=10, c = col, marker='o')
        color_b = plt.colorbar(p2d_ML)
        color_b.set_label('Profundidad (metros)')
        ay.set_title('Diagrama M vs L')
        plt.show()

        ##NL
        P_N  = [0.5241,0.5848, 0.6273, 0.5241]
        P_L  = [1.4847, 1.414, 1.2898, 1.4847]
        P__N = [a_x,    c_x,     d_x,     a_x]
        P__L = [a_z,    c_z,     d_z,     a_z]
        #coli = np.linspace(1,99,400)
        fig = plt.figure()
        az = fig.add_subplot(111)
        #az.title("Grafico N vs L")
        az.plot(P__N,P__L)
        az.plot(P_N,P_L)
        az.grid()
        az.set_xlabel('N')
        az.set_ylabel('L')
        az.scatter(N, L, s=10, c = col, marker='o')
        p2d_NL = az.scatter(N, L, s=10, c = col, marker='o')
        color_b = plt.colorbar(p2d_NL)
        color_b.set_label('Profundidad (metros)')
        az.set_title('Diagrama N vs L')
        plt.show()





if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
