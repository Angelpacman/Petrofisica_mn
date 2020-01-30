import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PyQt5 import uic, QtWidgets

qtCreatorFile = "petrofisica.ui" # Nombre del archivo aquí.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

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

        PROF= np.array(self.datos['PROF'])  #*-1
        col = np.linspace(-1*PROF[0],-1*PROF[-1],400)

        #aqui va figura en 2D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
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



        PROF= np.array(self.datos['PROF'])  #*-1
        col = np.linspace(-1*PROF[0],-1*PROF[-1],400)

        #aqui va figura en 2D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(P_inicial,P_final,P_M1,P_M2,v_x1,v_y1,v_x2,v_y2,v_x3,v_y3)
        ax.grid()
        ax.set_xlabel('N')
        ax.set_ylabel('M')
        ax.scatter(N, M, s=10, c = col, marker='o')
        p2d = ax.scatter(N, M, s=10, c = col, marker='o')
        color_b = plt.colorbar(p2d)
        color_b.set_label('Profundidad')
        ax.set_title('Diagrama M vs N')
        plt.show()


        ##ML
        P_M  =[0.7781,0.8269,0.8091,0.7781]
        P_L = [1.4847, 1.414, 1.2898, 1.4847]
        fig = plt.figure()
        ay = fig.add_subplot(111)
        #ay.title("Grafico M vs L")
        ay.plot(P_M,P_L)
        ay.grid()
        ay.set_xlabel('M')
        ay.set_ylabel('L')
        ay.scatter(M, L, s=10, c = col, marker='o')
        #p2d = ax.scatter(M, L, s=10, c = col, marker='o')
        color_b = plt.colorbar(p2d)
        color_b.set_label('Profundidad')
        ay.set_title('Diagrama M vs L')

        ##NL
        P_N = [0.5241,0.5848, 0.6273, 0.5241]
        P_L = [1.4847, 1.414, 1.2898, 1.4847]
        coli = np.linspace(1,99,400)
        fig = plt.figure()
        az = fig.add_subplot(111)
        #az.title("Grafico N vs L")
        az.plot(P_N,P_L)
        az.grid()
        az.set_xlabel('N')
        az.set_ylabel('L')
        az.scatter(N, L, s=10, c = col, marker='o')
        #p2d = az.scatter(M, L, s=10, c = coli, marker='o')
        color_b = plt.colorbar(p2d)
        color_b.set_label('Profundidad')
        az.set_title('Diagrama N vs L')
        plt.show()





if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
