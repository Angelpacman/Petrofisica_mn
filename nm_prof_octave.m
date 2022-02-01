% matlab file
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

%vector_prof = [round(min(PROF)): 1:round(max(PROF))];
%c = linspace(round(min(PROF)),round(max(PROF)),length(PROF));

%% Grafica de MN vs PROF Scatterplot en 3D
scatter3(N,M,PROF,20,PROF,'filled'); 
ax = gca; view(-31,14), xlabel('N'), ylabel('M'), zlabel('Profundidad (metros)');%,  cb = colorbar(gca);
colorbar();
%cb.Label.String = 'Profundidad del dato';
