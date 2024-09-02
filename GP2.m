%% Fits the numerical analysis-informed GP for multivariate output
%
% Inputs:
% fnt = (n1 x n2) array of data f(x_i,t_j)
% Xn1 = (n1 x d) array of discretisation parameters x_i in R^d
% Xn2 = (n2 x 1) vector of discretisation parameters t_j in R
% b1  = function handle b1(x) to error bound b(x), assumed t-independent
% ke1 = function handle ke1(x,y) to kernel with x,y in R^d
% ke2 = function handle ke2(x,y) to kernel with x,y in R
%
% Outputs:
% mn = (R^d x R -> R) mean function for the fitted Gaussian process model
% kn = ((R^d x R) x (R^d x R) -> R) covariance function for the fitted Gaussian process model

function [mn,kn] = GP2(fnt,Xn1,Xn2,b1,ke1,ke2)

% dimensions
[n1,~] = size(Xn1);
[n2,~] = size(Xn2);

% linearly order the data
tmp = fnt';
lin_fnt = tmp(:);

% sigma^2 not estimated, as not used
sig2 = 1;

% kb kernel
kb = @(X1,Y1) diag(b1(X1)) * ke1(X1,Y1) * diag(b1(Y1));

% conditional mean and covariance
mn = @(X1,x2) kron( kb(X1,Xn1) * inv(kb(Xn1,Xn1)) ...
                    + (1 - kb(X1,Xn1) *(kb(Xn1,Xn1) \ ones(n1,1))) ...
                      * ones(1,n1) * inv(kb(Xn1,Xn1)) ...
                      / (ones(1,n1) * (kb(Xn1,Xn1) \ ones(n1,1))) , ...
                    ke2(x2,Xn2) * inv(ke2(Xn2,Xn2)) ) ...
              * lin_fnt;
kn = @(X1,x2,Y1,y2) sig2 * ( kb(X1,Y1) * ke2(x2,y2) ...
                             - ( ke2(x2,Xn2)*(ke2(Xn2,Xn2)\ke2(Xn2,y2)) ) ...
                               * ( kb(X1,Xn1)*(kb(Xn1,Xn1)\kb(Xn1,Y1)) - (kb(X1,Xn1)*(kb(Xn1,Xn1)\ones(n1,1))-1) ...
                                                                         * (kb(X1,Xn1)*(kb(Xn1,Xn1)\ones(n1,1))-1)' ...
                                                                         / (ones(1,n1)*(kb(Xn1,Xn1)\ones(n1,1))) ) );
% return essentially infinity if x2 or y2 not in training set
kn = @(X1,x2,Y1,y2) kn(X1,x2,Y1,y2) + (1-ismember(x2,Xn2)) * (1-ismember(y2,Xn2)) * (1/eps); 

end




















