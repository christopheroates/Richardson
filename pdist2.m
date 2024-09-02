%% Pairwise distance matrix
% As with the in-built function pdist2, but rewritten in an elementary way 
% to allow variable precision arithmetic to be used

function out = pdist2(X,Y)

[nX,d] = size(X);
[nY,d] = size(Y);

X = repmat(X,nY,1);
Y = repelem(Y,nX,1);
tmp = mean((X-Y).^2,2).^(1/2);
out = reshape(tmp,nX,nY);

end