# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'petrofisica.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(784, 600)
        MainWindow.setStyleSheet("background-color: rgb(16, 38, 70);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 430, 431, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(310, 10, 241, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 470, 241, 51))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 530, 241, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(80, 20, 81, 101))
        self.label_5.setStyleSheet("border-image: url(:/poli/ipn.jpg);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(610, 20, 111, 101))
        self.label_6.setStyleSheet("border-image: url(:/esia/esia.png);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.aguassomeras = QtWidgets.QPushButton(self.centralwidget)
        self.aguassomeras.setGeometry(QtCore.QRect(320, 250, 141, 61))
        self.aguassomeras.setStyleSheet("background-color: rgb(140, 140, 140);")
        self.aguassomeras.setObjectName("aguassomeras")
        self.oleaje = QtWidgets.QPushButton(self.centralwidget)
        self.oleaje.setGeometry(QtCore.QRect(120, 250, 141, 61))
        self.oleaje.setStyleSheet("background-color: rgb(140, 140, 140);")
        self.oleaje.setObjectName("oleaje")
        self.lorenz = QtWidgets.QPushButton(self.centralwidget)
        self.lorenz.setGeometry(QtCore.QRect(520, 250, 141, 61))
        self.lorenz.setStyleSheet("background-color: rgb(140, 140, 140);")
        self.lorenz.setObjectName("lorenz")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 784, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#00ffff;\">Herramienta de visualización para evaluación petrofísica version 1.0.6</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#1cf0e9;\">Evaluacíón petrofísica</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#00ffff;\">Autor: RESENDIZ AVILÉS JOSE ANGEL</span></p><p><span style=\" color:#00ffff;\">angelr4a1@gmail.com</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#00ffff;\">Ascesor:</span></p></body></html>"))
        self.aguassomeras.setText(_translate("MainWindow", "Aguas Someras 3D"))
        self.oleaje.setText(_translate("MainWindow", "Oleaje 2D"))
        self.lorenz.setText(_translate("MainWindow", "Atractor de Lorenz"))
import logoesia_rc
import logopoli_rc
