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

M = 0.01 .* (189 - DT)./(RHOB - 1);
N = (1 - NPHI) ./ (RHOB - 1);
L = 0.01 .* (189 - DT)./(1 - NPHI);

scatter3(N,M,L,30,PROF,'filled'); 
ax = gca; view(-31,14), xlabel('N'), ylabel('M'), zlabel('L'),  cb = colorbar;
cb.Label.String = 'Profundidad del dato';