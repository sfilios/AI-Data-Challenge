
%% Initialization
%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
% Read in the data
data = csvread('wheat-2013-supervised.csv');
X = data(2:end, 6:18); y = data(2:end, 19); %first row is metadata, skip the location names to avoid overfitting
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
X(1:10,:) 
y(1:10,:)

fprintf('Program paused. Press enter to continue.\n');
pause;

% Normalize the features
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(14, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

