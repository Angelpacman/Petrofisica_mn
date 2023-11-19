# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# AUTHOR: JOSE ANGEL RESENDIZ AVILES
#
# PROJECT CREATED WITH:  PySide6, openpyxl, scipy, matplotlib, numpy
#
# This project has MIT license and is intented to provide a Graphical
# use interface to have insigts and a graphical aproach to geophysical 
# well logs unto M, N and L litologic parameters. 
#
# Github profile: @Angelpacman
#
# v2.0.1
# 2019-2023
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# IMPORT QT CORE
from qt_core import *

# MAIN WINDOW
class UI_MainWindow(object):
    def setup_ui(self, parent):
        if not parent.objectName():
            parent.setObjectName("MainWindow")

        # SET INITIAL PARAMETERS
        parent.resize(920,600)
        parent.setMinimumSize(720, 480) 

        # CREATE CENTRAL WIDGET
        self.central_frame = QFrame()
        #self.central_frame.setStyleSheet("background-color: #282a36")

        # SET CENTRAL WIDGET
        parent.setCentralWidget(self.central_frame)                 #Definir lienzo prinicipal

        # CREAR MAIN LAYOUT
        self.main_layout = QHBoxLayout(self.central_frame)          # Ocupar el lienzo
        self.main_layout.setContentsMargins(0,0,0,0)                # Desagregar los bordes del lienzo
        self.main_layout.setSpacing(0)                              # Quitar espacio entre los contenidos

        # LEFT MENU
        self.left_menu = QFrame()                                   # Crear barra con Qframe
        self.left_menu.setStyleSheet("background-color: #44475a")   # Definir color de la barra

        # CONTENIDO
        self.content = QFrame()                                     # Crear cuadro de contenido con Qframe
        self.content.setStyleSheet("background-color: #282a36")     # Definir color del cuadro

        # ADD WIDGETS TO APP
        self.main_layout.addWidget(self.left_menu)                  # Agregar los Qframe al lienzo, se
        self.main_layout.addWidget(self.content)                    #ordenan horizontalmente de izq a der

        # SET CENTRAL WIDGET
        parent.setCentralWidget(self.central_frame)                 # Se ocupa la ventana con la definicion de lienzo