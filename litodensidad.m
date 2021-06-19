clear; close all; clc;

datos = readtable('eval_petro(0).csv');
PROF = -1.*datos.PROF;
LLD = datos.LLD;
LLS = datos.LLS;
FR = datos.FR;
DT = datos.DT;
NPHI = datos.NPHI;
RHOB = datos.RHOB;
GR = datos.GR;

%% Aqui vienen mis definiciones de las varibles M,N,L
M = 0.01 .* (189 - DT)./(RHOB - 1);
N = (1 - NPHI) ./ (RHOB - 1);
L = 0.01 .* (189 - DT)./(1 - NPHI);

%% Ahora el cálculo de los valores ideales Dol, Cal, Sil
DOLOMIA = [43.5,   2.87,   0.02];
CALIZA  = [47.6,   2.71,   0.00];
SILICE  = [55.5,   2.65,  -0.035];
ARCILLA = [120,    2.35,   0.33];


%% Grafica de MN vs PROF Scatterplot en 3D
scatter3(N,M,PROF,30,PROF,'filled'); 
ax = gca; view(-31,14), xlabel('N'), ylabel('M'), zlabel('Profundidad en metros'),  cb = colorbar;
cb.Label.String = 'Profundidad del dato';


