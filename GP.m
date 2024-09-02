%% Fits the numerical analysis-informed GP for univariate output
%
% Inputs:
% fn = (n x 1) vector of data f(x_i)
% Xn = (n x d) array of discretisation parameters x_i in R^d
% b  = function handle b(x) to error bound
% ke = function handle ke(x,y) to kernel
%
% Outputs:
% mn = (R^d -> R) mean function for the fitted Gaussian process model
% kn = (R^d x R^d -> R) covariance function for the fitted Gaussian process model
% nlq1 = negative (2x) log quasi likelihood, up to additive constant
% sig2 = estimated amplitude sigma_n[f], squared

function [mn,kn,nlql,sig2] = GP(fn,Xn,b,ke)

% dimensions
[n,d] = size(Xn);

% building blocks
A1 = ones(n,1) ./ b(Xn);
A2 = fn .* A1;
A3 = inv(ke(Xn,Xn)); 
A4 = (A1' * A3 * A2) / (A1' * A3 * A1);
A5 = (1/n) * (A2' * A3 * A2 - (A1' * A3 * A2)^2 / (A1' * A3 * A1)); 

% error handling
A5 = max(eps,A5); % avoid machine 0

% conditional mean and covariance
mn = @(X) A4 ...
          + b(X) .* ke(X,Xn) * A3 * A2 ...
          - b(X) .* ke(X,Xn) * A3 * A1 * A4;
kn = @(X,Y) A5 * ( b(X) .* ke(X,Y) .* (b(Y)') ...
                   - b(X) .* ke(X,Xn) * A3 * (ke(Xn,Y) .* (b(Y)')) ...
                   + (b(X) .* ke(X,Xn) * A3 * A1 - ones(size(X,1),1)) ...
                     * ((b(Y) .* ke(Y,Xn) * A3 * A1 - ones(size(Y,1),1))') ...
                     / (A1' * A3 * A1) );

% negative (2x) log quasi likelihood, up to additive constant
%nlql = n * A5 + 2 * sum(log(b(Xn))) + log(det(ke(Xn,Xn))); % constant sigma^2 used
nlql = n * log(A5) + 2 * sum(log(b(Xn))) + log(det(ke(Xn,Xn))); % plug-in sigma^2 used

% estimated sigma^2
sig2 = A5;

% error handling
kn = @(X,Y) max(0,kn(X,Y));

end




















