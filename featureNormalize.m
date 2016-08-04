function [X_norm, mu, sigma] = featureNormalize(X)

%take the mean of each feature and normalize by taking the mean of each feature and dividing by standard deviation

mu = mean(X);
sigma = std(X);
m=size (X,1);
mu_matrix = ones(m, 1) * mu;
sigma_matrix = ones(m,1)*sigma;
X_norm=X-mu_matrix;
X_norm=X./sigma_matrix;


end
