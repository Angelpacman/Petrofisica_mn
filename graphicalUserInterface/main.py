
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


# IMPORT MODULES
import sys
import os

# IMPORT QT CORE
from qt_core import *

# IMPORT MAIN WINDOW
from gui.windows.main_window.ui_main_window import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # SETUP MAIN WINDOW
        self.ui = UI_MainWindow()
        self.ui.setup_ui(self)

        # SHOW THE APPLICATION
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
    
