%% calculateDamageIndex.m
%% Integrated version for calculating the damage index using Delay-and-Sum with multiple grid sizes
clear; clc; close all;
format longE

%% General parameters
num_sensors = 12;
group_velocity = 5000;
N = 4096;
fs = 12.5E6;
Ts = 1 / fs;
dtb = 0;
t = 0:Ts:(N-1)*Ts;

%% Plate parameters
width = 0.3;   
height = 0.7; 

%% Grid sizes to test
grid_sizes = [75, 120, 200, 300];

results_filename = 'test_results.txt';
fid = fopen(results_filename, 'w');
fprintf(fid, 'Test,Grid_X,Grid_Y,MeanTime_ms,StdTime_ms\n');
fclose(fid);

%% Run tests for each grid size
for test = 1:length(grid_sizes)
    num_points_x = grid_sizes(test);
    num_points_y = grid_sizes(test);
    
    fprintf("\n=============================\n");
    fprintf("Test %d - Grid size: %dx%d\n", test, num_points_x, num_points_y);
    fprintf("=============================\n");
    
    [x_grid, y_grid, distances] = createGrid(num_points_x, num_points_y, num_sensors, width, height);

    %% Repeat algorithm 1000 times and measure execution time
    num_iterations = 1;
    execution_times = zeros(1, num_iterations);

    for iter = 1:num_iterations
        tic;
        %% Load data
        data_damage = load('datadamage_scaled.mat');
        data_non_damage = load('datanondamage_scaled.mat');
        
        %% Compute Hilbert transforms once
        hilbert_damage = computeHilbert(data_damage.data, num_sensors, N);
        hilbert_non_damage = computeHilbert(data_non_damage.data, num_sensors, N);
        
        %% Initialize output matrices
        damage_index = zeros(num_points_y, num_points_x);
        non_damage_index = zeros(num_points_y, num_points_x);
        damage_index_diff = zeros(num_points_y, num_points_x);
        
        %% Main Delay-and-Sum algorithm
        for row = 1:num_points_y
            for col = 1:num_points_x
                sum_damage = 0;
                sum_non_damage = 0;
                sum_diff = 0;
                for e = 1:num_sensors
                    for r = 1:num_sensors
                        if e ~= r
                            d_e = distances(row, col, e);
                            d_r = distances(row, col, r);
                            n0 = ceil(((d_e + d_r) / group_velocity + dtb) / Ts);
                            if n0 <= N
                            signal_damage = data_damage.data(n0, e, r) + 1i * hilbert_damage{e, r}(n0);
                            signal_non_damage = data_non_damage.data(n0, e, r) + 1i * hilbert_non_damage{e, r}(n0);
                            signal_diff = signal_non_damage - signal_damage;

                            aux = sqrt(d_e * d_r);
                            sum_damage = sum_damage + aux * signal_damage;
                            sum_non_damage = sum_non_damage + aux * signal_non_damage;
                            sum_diff = sum_diff + aux * signal_diff;
                            end
                        end
                    end
                end
                damage_index(row, col) = abs(sum_damage);
                non_damage_index(row, col) = abs(sum_non_damage);
                damage_index_diff(row, col) = abs(sum_diff);
            end
        end
        execution_times(iter) = toc;
    end

    %% Calculate statistics
    mean_time = mean(execution_times)*1000;
    std_time = std(execution_times)*1000;

    fprintf("Tiempo medio de ejecución: %.8f ms\n", mean_time);
    fprintf("Desviación típica: %.8f ms\n", std_time);

    %% Guarda los resultados en el archivo CSV (COMENTADO)
    fid = fopen(results_filename, 'a');
    fprintf(fid, '%d,%d,%d,%.8f,%.8f\n', test, num_points_x, num_points_y, mean_time, std_time);
    fclose(fid);

    %% Visualization of the last iteration
    figure;
    imagesc(x_grid(1, :), y_grid(:, 1), damage_index); colorbar;
    title(sprintf('Damage Index - With Damage (%dx%d)', num_points_x, num_points_y)); 
    axis xy;

    figure;
    imagesc(x_grid(1, :), y_grid(:, 1), non_damage_index); colorbar;
    title(sprintf('Damage Index - Without Damage (%dx%d)', num_points_x, num_points_y)); 
    axis xy;

    figure;
    imagesc(x_grid(1, :), y_grid(:, 1), damage_index_diff); colorbar;
    title(sprintf('Damage Index - Difference (%dx%d)', num_points_x, num_points_y)); 
    axis xy;

    %% Export results of the last iteration
    writematrix(damage_index, sprintf('damage_index_matlab_%dx%d.csv', num_points_x, num_points_y));
    writematrix(non_damage_index, sprintf('non_damage_index_matlab_%dx%d.csv', num_points_x, num_points_y));
    writematrix(damage_index_diff, sprintf('damage_index_diff_matlab_%dx%d.csv', num_points_x, num_points_y));
end

%% Function to create grid and compute distances
function [x_grid, y_grid, distances] = createGrid(num_points_x, num_points_y, num_sensors, width, height)
[x_grid, y_grid] = meshgrid( ...
    linspace(-0.2, width, num_points_x), ...
    linspace(-0.05, height, num_points_y) ...
);
sensor_pos_x = [0, 0.009, 0.018, 0.027, 0.036, 0.045, 0.054, 0.063, 0.072, 0.081, 0.090, 0.099];
sensor_pos_y = zeros(1, num_sensors);
distances = zeros(num_points_y, num_points_x, num_sensors);
for row = 1:num_points_y
    for col = 1:num_points_x
        for k = 1:num_sensors
            dx = x_grid(row, col) - sensor_pos_x(k);
            dy = y_grid(row, col) - sensor_pos_y(k);
            distances(row, col, k) = sqrt(dx^2 + dy^2);
        end
    end
end
end

%% Function to compute Hilbert transform array
function hilbertArray = computeHilbert(data, num_sensors, N)
hilbertArray = cell(num_sensors, num_sensors);
for e = 1:num_sensors
    for r = 1:num_sensors
        if e ~= r
            signal = data(1:N, e, r);
            hilbertArray{e, r} = real(hilbert_transform(signal));
        end
    end
end
end

%% Custom Hilbert transform implementation
function H = hilbert_transform(X)
Y = fft(X);
Y(1) = 0;
N = length(Y);
if mod(N, 2) == 0
    Y(N/2 + 1) = 0;
end
H = zeros(size(Y));
for k = 2:N
    if k <= N/2
        H(k) = -1i * Y(k);
    else
        H(k) = 1i * Y(k);
    end
end
H = ifft(H);
end
