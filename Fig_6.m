%% Reproduces cardiac experiment with scalar quantities of interest

clear all

% import dataset
data = cardiac_import_dataset();

% gold standard
dx_gold = 0.4;
dt_gold = 1;
ix_gold = ((data.X(:,1) == dx_gold) & (data.X(:,2) == dt_gold));

% lofi resolutions 
dx_lofi = 1.7; % dx for lofi
dt_lofi = 5; % dt for lofi
ix_dx_lofi = (data.X(:,1) == dx_lofi);
ix_dt_lofi = (data.X(:,2) == dt_lofi);
ix_lofi = (ix_dx_lofi | ix_dt_lofi);

% specify permissible data from which GRE is trained
dx_GRE_max = 0.7; % maximum dx for inclusion in GRE
dt_GRE_max = 4; % maximum dt for inclusion in GRE
ix_GRE = (data.X(:,1) <= dx_GRE_max) & (data.X(:,2) <= dt_GRE_max);
ix_GRE(ix_gold) = false;

% default resolutions 
dx_default = 0.4;
dt_default = 2;
ix_default = ((data.X(:,1) <= dx_default) & (data.X(:,2) <= dt_default)) & (~ix_gold);

% computational budget values
C_vals = [10000,30000,50000,100000,200000,1000000];

% plotting
figure(1)
set(gcf,'Position',[100,100,400,400])
clf
fac_x = 1.1;
fac_t = 1.6;
dx_max = max(data.X(:,1));
dt_max = max(data.X(:,2));
for dx = unique(data.X(:,1))
    plot([dx,dx],[0,fac_t*dt_max],'k:'); hold on;
end
for dt = unique(data.X(:,2))
    plot([0,fac_x*dx_max],[dt,dt],'k:')
end
xlabel("$x_1$ (millimeters)")
ylabel("$x_2$ (milliseconds)")
xlim([0,fac_x*dx_max])
ylim([0,fac_t*dt_max])
h0 = plot(data.X(ix_GRE,1),data.X(ix_GRE,2),'k.');
h1 = plot(dx_lofi,dt_lofi,'ko','MarkerFaceColor','k');
h2 = plot(dx_lofi,data.X(ix_dx_lofi,2),'ko');
plot(data.X(ix_dt_lofi,1),dt_lofi,'ko')

% scan over all scalar outputs
for p = 1:length(data.names)

    % univariate Gauss--Richardson Extrapolation
    r = [0.5;1;2];
    s = [0;1;2];
    [mnx,knx,bx_opt,kx_opt,sig2x_opt,r_opt(p,1),s_opt(p,1),l_opt(p,1)] = GRE(data.f(ix_dt_lofi,p),data.X(ix_dt_lofi,1),r,s);
    [mnt,knt,bt_opt,kt_opt,sig2t_opt,r_opt(p,2),s_opt(p,2),l_opt(p,2)] = GRE(data.f(ix_dx_lofi,p),data.X(ix_dx_lofi,2),r,s);

    % composite kernel
    k = @(X,Y) kx_opt(X(:,1),Y(:,1)) .* kt_opt(X(:,2),Y(:,2));

    % composite bound
    wx(p) = sig2x_opt^(1/2);
    wt(p) = sig2t_opt^(1/2);
    b = @(X) (wx(p)/(wx(p)+wt(p))) * bx_opt(X(:,1)) + (wt(p)/(wx(p)+wt(p))) * bt_opt(X(:,2));

    % approximate the cost function
    sf = fit([log(data.X(:,1)),log(data.X(:,2))],log(data.times),'poly11'); % assumes 1/polynomial growth
    c = @(X) exp(sf(log(X(:,1)),log(X(:,2))));
    %plot(sf,[log(data.X(:,1)),log(data.X(:,2))],log(data.times))

    % option to add in the lofi training data
    add_lofi = false;

    % experimental design
    XN = data.X(ix_GRE,:); % candidate states
    d = 2;
    min_num_points = 0;
    max_num_points = inf;
    constraint_type = 'sum';

    % loop through budget values
    for cix = 1:length(C_vals)

        % budget
        C = C_vals(cix);

        % experimental design
        [X_opt,cost] = design(k,b,c,C,XN,min_num_points,max_num_points,constraint_type);
        ix_opt = knnsearch(data.X,X_opt); 
    
        % add in the lofi training data if required
        ix_train = false(size(data.X,1),1);
        ix_train(ix_opt) = true;
        ix_train(ix_lofi) = add_lofi;
    
        % plotting
        if (p == 1) && (C == 100000)
            figure(p)
            h3 = plot(data.X(ix_opt,1),data.X(ix_opt,2),'k^');
            h4 = plot(data.X(ix_default,1),data.X(ix_default,2),'k^','MarkerFaceColor','k');
            h5 = plot(dx_gold,dt_gold,'kpentagram','MarkerFaceColor','k');
            legend([h5(1),h1(1),h4(1),h0(1),h2(1),h3(1)], ...
                   {"{\boldmath$x$}${}_{\mathrm{hi-fi}}$", ...
                    "{\boldmath$x$}${}_{\mathrm{lo-fi}}$", ...
                    "{\boldmath$x$}${}_{\mathrm{default}}$", ...
                    "candidate experiments", ...
                    "Step 1 - lo-fi experiments", ...
                    "Step 2 - experimental design"})
            title("GRE Workflow ($C = 10^5$ seconds)")
            print('workflow')
        end
    
        % multivariate Gauss--Richardson Extrapolation
        [mn,kn,nlql] = GP(data.f(ix_train,p),data.X(ix_train,:),b,k);
        x_hifi = [dx_gold,dt_gold];
    
        % store approximations and ground truth
        f_hifi(p) = data.f(ix_gold,p); % truth at x_hifi
        mn_hifi(cix,p) = double(mn(x_hifi)); % posterior mean
        std_hifi(cix,p) = double(sqrt(kn(x_hifi,x_hifi))); % posterior standard deviation
        f_default(:,p) = data.f(ix_default,p); % default resolution
    
    end

end

% typical values encountered (for use later in analysing temporal output)
disp("Convergence order in x1 = " + num2str(median(r_opt(:,1))))
disp("Convergence order in x2 = " + num2str(median(r_opt(:,2))))
disp("Smoothness in x1 = " + num2str(median(s_opt(:,1))))
disp("Smoothness in x2 = " + num2str(median(s_opt(:,2))))
disp("Lengthscale in x1 = " + num2str(mean(l_opt(:,1))))
disp("Lengthscale in x2 = " + num2str(mean(l_opt(:,2))))
disp("Error weight in x1 = " + num2str(mean(double(wx))))
disp("Error weight in x2 = " + num2str(mean(double(wt))))

% plotting
figure(8)
set(gcf,'Position',[100,100,400,400])
marker = {'k-o','k-^','k-pentagram','k-*','k-square','k-diamond','k-v'};
for p = 1:length(data.names)
    h{p} = loglog(C_vals,abs(mn_hifi(:,p) - f_hifi(p)) ...
                         ./abs(f_default(:,p) - f_hifi(p)),marker{p}); hold on;
end
yline(1,'k:')
xlabel('$C$ (seconds)')
ylabel('Abs Error Relative to $f(${\boldmath$x$}${}_{\mathrm{default}})$')
legend(data.names)
print('scalar')


