%% Reproduces cardiac experiment with temporal quantities of interest

clear all

% import dataset
data = cardiac_import_dataset();

% gold standard
dx_gold = 0.4;
dt_gold = 1;
ix_gold = ((data.X(:,1) == dx_gold) & (data.X(:,2) == dt_gold));

% specify permissible data from which GRE is trained
dx_GRE_max = 0.7; % maximum dx for inclusion in GRE
dt_GRE_max = 4; % maximum dt for inclusion in GRE
ix_GRE = (data.X(:,1) <= dx_GRE_max) & (data.X(:,2) <= dt_GRE_max);
ix_GRE(ix_gold) = false;

% default resolutions 
dx_default = 0.4;
dt_default = 2;
ix_default = ((data.X(:,1) <= dx_default) & (data.X(:,2) <= dt_default)) & (~ix_gold);

% composite kernel - based on average of kernels used for scalar qois
lx = 1.7857; % average from scalar case 
lt = 16.6; % average from scalar case                                                
kex = @(X,Y) exp( - pdist2(X./lx,Y./lx) ); % Matern 1/2    
ket = @(X,Y) (1 + sqrt(3) * pdist2(X./lt,Y./lt)) ...
              .* exp( - sqrt(3) * pdist2(X./lt,Y./lt) ); % Matern 3/2 
ke = @(X1,Y1) kex(X1(:,1),Y1(:,1)) .* ket(X1(:,2),Y1(:,2));     

% composite bound - based on average of kernels used for scalar qois
wx = 2.8401; % average from scalar case                                      
wt = 0.59463; % average from scalar case                                      
bx = @(X1) X1;
bt = @(X2) X2;
b = @(X1) (wx/(wx+wt)) * bx(X1(:,1)) + (wt/(wx+wt)) * bt(X1(:,2));

% covariance for t index
ell_t = 600; % length of the overall time series                                    
kt = @(X,Y) exp( - pdist2(X/ell_t,Y/ell_t) );

% approximate the cost function
sf = fit([log(data.X(:,1)),log(data.X(:,2))],log(data.times),'poly11'); % assumes 1/polynomial growth
c = @(X) exp(sf(log(X(:,1)),log(X(:,2))));
%plot(sf,[log(data.X(:,1)),log(data.X(:,2))],log(data.times))

% scan over all temporal outputs
names = {'Left Atrium','Left Ventricle','Right Atrium','Right Ventrical'};
figure(1)
set(gcf,'Position',[100,100,800,400])
clf
for p = 1:4

    % multivariate Gauss--Richardson Extrapolation
    [mn,kn] = GP2(data.ft(ix_GRE,:,p),data.X(ix_GRE,:),data.tgrid,b,ke,kt);
    x_hifi = [dx_gold,dt_gold];

    % store approximations and ground truth
    f_hifi(:,p) = data.ft(ix_gold,:,p); % truth at x_hifi
    f_default(:,p) = data.ft(ix_default,:,p); % truth at x_default
    mn_hifi(:,p) = double(mn(x_hifi,data.tgrid)); % posterior mean

    % relative mean square error
    rmse(p) = mean((f_hifi(:,p) - mn_hifi(:,p)).^2) ...
              / mean((f_hifi(:,p) - f_default(:,p)).^2);
    
    % plotting
    subplot(3,2,p)
    T = length(data.tgrid);
    h1 = plot(data.tgrid,data.ft(ix_GRE,:,p),'k:'); hold on;
    h2 = plot(data.tgrid,f_hifi(:,p),'k-'); 
    h3 = plot(data.tgrid,f_default(:,p),'b^:','MarkerIndices',1:T/10:T); 
    h4 = plot(data.tgrid,mn_hifi(:,p),'ro-','MarkerIndices',1:T/10:T);
    if p > 2
        xlabel('$t$')
    else
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
    end
    ylabel(names{p})
    title(['Relative MSE = ',num2str(rmse(p),3)])
    if p == 4
        legend([h2(1),h1(1),h3(1),h4(1)], ...
               {'$f(${\boldmath$x$}${}_{\mathrm{hi-fi}},t)$', ...
                'training data', ...
                '$f(${\boldmath$x$}${}_{\mathrm{default}},t)$', ...
                '$m_n[f](${\boldmath$x$}${}_{\mathrm{hi-fi}},t)$' }, ...
                'Orientation','horizontal', ...
                'fontsize',10, ...
                'location','southoutside','position',[0.31,0.3,0.58,0.03]);  % left bottom width height
    end

    % inset plot
    pos = get(gca,'Position'); % left bottom width height
    pos = [pos(1) + 0.7 * pos(3), ...
           pos(2) + 0.3 * pos(4), ...
           0.28 * pos(3), ...
           0.6 * pos(4)];
    lwid = 1;
    showAxisLabels = false;
    t_range = [240,250];
    t_box_range = [230,260];
    f_range = [min(min(data.ft(ix_GRE,t_range,p))), ...
               max(max(data.ft(ix_GRE,t_range,p)))];
    magnifyPlot(t_box_range,f_range,pos,lwid,showAxisLabels)

end

print('temporal')



