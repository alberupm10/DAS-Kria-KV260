clear, clc, close all

%% Cargar datos de entrada
%%data = load('datanondamage.mat');
data = load('datadamage.mat');

%% Definir factor de escala
scale_factor = 100; % Multiplica por 1000 para aumentar la magnitud de los valores

%% Aplicar el factor de escala
data = data.data * scale_factor;
%%data = data.data * scale_factor;

%% Guardar los datos escalados en nuevos archivos .mat
save('datadamage_scaled.mat', 'data');
%%save('datanondamage_scaled.mat', 'data');

disp('Datos escalados guardados en "datadamage_scaled.mat" y "datanondamage_scaled.mat"');
