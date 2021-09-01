% octave file

clear; close all; clc;
pkg load io

datos = xlsread('eval_petro(0).xlsx');

PROF = -1.*datos(:,1);
LLD = datos(:,3);
LLS = datos(:,4);
FR = datos(:,5);
DT = datos(:,6);
NPHI = datos(:,7);
RHOB = datos(:,8);
GR = datos(:,2);

M = 0.01 .* (189 - DT)./(RHOB - 1);
N = (1 - NPHI) ./ (RHOB - 1);
L = 0.01 .* (189 - DT)./(1 - NPHI);

scatter3(N,M,L,30,PROF,'filled'); 
ax = gca; view(-31,14), xlabel('N'), ylabel('M'), zlabel('L'),  cb = colorbar;
cb.Label.String = 'Profundidad del dato';