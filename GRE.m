%% Gauss--Richardson Extrapolation for univariate output
%
% Uses maximum quasi likelihood to estimate an appropriate Gaussian process
% multi-fidelity model.
%
% Inputs:
% fn       = (n x 1) vector of data f(x_i)
% Xn       = (n x d) array of discretisation parameters x_i in R^d
% r        = (dr x d) array whose rows are candidate multi-indicies ri for b(x) = x^ri
% s        = (ds x 1) array whose entries are candidate smoothnesses for the normalised error
% varargin = (1 x nl) vector of candidate length-scale parameters for the kernel 
%
% Outputs:
% mn       = (R^d -> R) mean function for the fitted Gaussian process model
% kn       = (R^d x R^d -> R) covariance function for the fitted Gaussian process model
% b_opt    = (R^d -> R) fitted error bound
% ke_opt   = (R^d x R^d -> R) fitted kernel
% sig2_opt = estimated amplitude sigma_n[f], squared
% r_opt    = (1 x d) array whose rows are multi-indicies ri for b(x) = x^ri
% s_opt    = estimated smoothness
% l_opt    = estimate length-scale

function [mn,kn,b_opt,ke_opt,sig2_opt,r_opt,s_opt,l_opt] = GRE(fn,Xn,r,s,varargin)

%% data normalisation
Xn_normalised = Xn ./ max(Xn,[],1);
[~,d] = size(Xn);

%% enumerate models
digits(30) % high-precision arithmetic
if nargin == 5
    ells = varargin{1};
else
    ells = vpa([0.1,0.2,0.5,1,2,5,10,15,20,100]); % candidate kernel length-scales (for normalised Xn)
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

%% assess models
for m_ix = 1:model_counter
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
    [~,~,nlql(m_ix)] = GP(fn,Xn_normalised,b{m_ix},ke{m_ix});    
end

%% find best model
[~,m_ix] = min(nlql); 
b_opt = b{m_ix};
ke_opt = ke{m_ix};
r_ix = model{m_ix}(1);
s_ix = model{m_ix}(2);
l_ix = model{m_ix}(3);
r_opt = r(r_ix,:);
s_opt = s(s_ix);
l_opt = double(vec_ells(l_ix,:));
[mn,kn,~,sig2_opt] = GP(fn,Xn_normalised,b{m_ix},ke{m_ix});

%% return to original data scale
mn = @(X) mn(X ./ max(Xn,[],1));
kn = @(X,Y) kn(X ./ max(Xn,[],1),Y ./ max(Xn,[],1));
b_opt = @(X) b_opt(X ./ max(Xn,[],1));
ke_opt = @(X,Y) ke_opt(X ./ max(Xn,[],1),Y ./ max(Xn,[],1));

end









