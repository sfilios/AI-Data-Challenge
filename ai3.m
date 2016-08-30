
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

%calculate theta with normal equation
theta=normalEqn.m

% Add intercept term to X
X = [ones(m, 1) X];


% Display normaleqn result
fprintf('Theta computed from normal equation: \n');
fprintf(' %f \n', theta);
fprintf('\n');


