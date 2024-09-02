%% Gauss--Richardson Extrapolation for multivariate output
%
% Uses maximum quasi likelihood to estimate an appropriate Gaussian process
% multi-fidelity model.
%
% Inputs:
% fn       = (n x T) matrix containing evaliations f(xi,:) of the multi-fidelity model f(x,t)
% Xn       = (n x d) array of discretisation parameters x_i in R^d
% tn       = (T x 1) vector of times
% r        = (dr x d) array whose rows are candidate multi-indicies ri for b(x) = x^ri
% s        = (ds x 1) array whose entries are candidate smoothnesses for the normalised error
% varargin = (1 x nl) vector of candidate length-scale parameters for the kernel 
%
% Outputs:
% mnt       = (R^d x R -> R) mean function for the fitted Gaussian process model
% knt       = ((R^d x R) x (R^d x R) -> R) covariance function for the fitted Gaussian process model
% b_opt    = (R^d -> R) fitted error bound
% ke_opt   = (R^d x R^d -> R) fitted kernel
% kt_opt   = (R x R -> R) fitted kernel

function [mnt,knt,b_opt,ke_opt,kt_opt] = GRE2(fnt,Xn,tn,r,s,varargin)

%% data normalisation
Xn_normalised = Xn ./ max(Xn,[],1);
[~,d] = size(Xn);

%% enumerate models
digits(30)
if nargin == 6
    ells = varargin{1};
else
    ells = vpa([0.1,0.2,0.5,1,2,5,10,15,20,100]); % candidate kernel length-scales
end
if d > 1
    PR = perms_rep(length(ells),d);
    vec_ells = ells(PR);
else
    vec_ells = ells';
end
model_counter = 0;
for r_ix = 1:size(r,1)
    for s_ix = 1:length(s)
        for l_ix = 1:size(vec_ells,1)
            model_counter = model_counter + 1;
            model{model_counter} = [r_ix,s_ix,l_ix];
        end
    end
end

%% covariance for t index
ell_t = range(tn);  
kt = @(X,Y) exp( - pdist2(X/ell_t,Y/ell_t) );

%% assess models
for m_ix = 1:model_counter
    %disp(["Fitting model ",num2str(m_ix,'%u')," of ",num2str(model_counter,'%u')])
    %drawnow
    r_ix = model{m_ix}(1);
    s_ix = model{m_ix}(2);
    l_ix = model{m_ix}(3);
    b{m_ix} = @(X) sum(X.^(r(r_ix,:)),2);
    l = vec_ells(l_ix,:);
    if s(s_ix) == 0
        ke{m_ix} = @(X,Y) exp( - pdist2(X./l,Y./l) ); % Matern 1/2
    elseif s(s_ix) == 1
        ke{m_ix} = @(X,Y) (1 + sqrt(3) * pdist2(X./l,Y./l)) ...
                   .* exp( - sqrt(3) * pdist2(X./l,Y./l) ); % Matern 3/2
        %ke{m_ix} = @(X,Y) max(zeros(size(X,1),size(Y,1)),1-pdist2(X./l,Y./l)).^3 ...
        %                  .* (3*pdist2(X./l,Y./l) + 1); % Wendland s = 1
    elseif s(s_ix) == 2
        ke{m_ix} = @(X,Y) (1 + sqrt(5) * pdist2(X./l,Y./l) + (5/3) * pdist2(X./l,Y./l).^2) ...
                   .* exp( - sqrt(5) * pdist2(X./l,Y./l) ); % Matern 5/2
        %ke{m_ix} = @(X,Y) max(zeros(size(X,1),size(Y,1)),1-pdist2(X./l,Y./l)).^5 ...
        %                  .* (8*pdist2(X./l,Y./l).^2 + 5*pdist2(X./l,Y./l) + 1); % Wendland s = 2
    elseif s(s_ix) == inf
        ke{m_ix} = @(X,Y) exp( - pdist2(X./l,Y./l).^2 ); % Gaussian
    end
    [~,~,nlql(m_ix)] = GP2(fnt,Xn_normalised,tn,b{m_ix},ke{m_ix},kt);    
end

%% find best model
[~,m_ix] = min(nlql); 
b_opt = b{m_ix};
ke_opt = ke{m_ix};
r_ix = model{m_ix}(1);
s_ix = model{m_ix}(2);
l_ix = model{m_ix}(3);
disp("Optimal r = " + num2str(r(r_ix,:)))
disp("Optimal s = " + num2str(s(s_ix)))
disp("Optimal ell = " + num2str(double(vec_ells(l_ix,:))))
[mnt,knt] = GP2(fnt,Xn_normalised,tn,b{m_ix},ke{m_ix},kt);

%% return to original data scale
mnt = @(X1,X2) mnt(X1 ./ max(Xn,[],1),X2);
knt = @(X1,X2,Y1,Y2) knt(X1 ./ max(Xn,[],1),X2,Y1 ./ max(Xn,[],1),Y2);
b_opt = @(X1) b_opt(X1 ./ max(Xn,[],1));
ke_opt = @(X1,Y1) ke_opt(X1 ./ max(Xn,[],1),Y1 ./ max(Xn,[],1));
kt_opt = @(X2,Y2) kt(X2,Y2);

end









