%% Univariate exratapolation methods
%
% Inputs:
% fn     = (n x 1) vector of data f(x_i)
% Xn     = (n x d) array of discretisation parameters x_i in R^d, ordered from
%          coarsest resolution (i = 1) to highest resolution (i = n)
% method = name of extrapolation method
%
% Output:
% out = the approximation produced by applying the extrapolation method

function out = univariate_extrapolation(fn,Xn,method)

[n,~] = size(Xn);

if strcmp(method,"Richardson")
    p = n;
    for i = 1:p-1
        for j = 1:p
            g(i,j) = Xn(j)^i;
        end
    end   
elseif strcmp(method,"Shanks")
    p = floor(n/2);
    for i = 1:p-1
        for j = 1:p
            g(i,j) = fn(i+j) - fn(i+j-1);
        end
    end
elseif strcmp(method,"Germain-Bonne")
    p = n-1;
    for i = 1:p-1
        for j = 1:p
            g(i,j) = (fn(j+1) - fn(j))^i;
        end
    end
elseif strcmp(method,"Thiele")
    q = floor((n-1)/2);
    p = 2*q - 1;
    for i = 1:q-1
        for j = 1:p
            g(i,j) = Xn(j)^i;
            g(i+(q-1),j) = fn(j) * Xn(j)^i;
        end
    end
end

log_out = logdet([fn(1:p)'; g]) - logdet([ones(1,p); g]);
out = exp(log_out);


end