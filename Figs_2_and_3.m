%% Reproduce examples from the main text

clear all

for application = 1:2

    %% simulated data
    if application == 1
    
        %% finite difference approxmation
        t0 = 0;
        a = 10;
        g = @(T) sin(a * T) + (T > 0) .* (T.^6); f0 = a; % exactly five times continuously differentiable
        
        f = @(X) (g(t0+X) - g(t0-X)) / (2*X); % centred difference approximation to g'(t0)
    
        n = 5;
        Xn = linspace(1/n,1,n)';
        
        % error bound
        b = @(X) X.^2; % (requires g to be thrice continuously differentiable)
    
    elseif application == 2
    
        %% trapezoidal rule
        % approximation of \int_{-1}^{1} g(t) dt
        a = 10;
        g = @(T) sin(a * T) + T.^2; f0 = (1/a) * (1 - cos(a)) + 1/3; % exactly four times continuously differentiable
        
        f = @(X) trapz(linspace(0,1,1/X)', ...
                       g(linspace(0,1,1/X)')); % centred difference approximation to g'(t0)
    
        n = 5;
        Xn = 2.*(1/2).^((2:(n+1))');
        
        % b function
        b = @(X) X.^2; % (requires g to be twice continuously differentiable)
    
    end
    
    % higher precision arithmetic
    digits(100)
    ell = vpa(1); % make length-scale higher precision to force all to be higher precision
    
    % kernels
    k{1} = @(X,Y) exp( - pdist2(X./ell,Y./ell) ); % Matern 1/2
    k{2} = @(X,Y) (1 + sqrt(3) * pdist2(X./ell,Y./ell)) .* exp( - sqrt(3) * pdist2(X./ell,Y./ell) ); % Matern 3/2
    k{3} = @(X,Y) (1 + sqrt(5) * pdist2(X./ell,Y./ell) + (5/3) * pdist2(X./ell,Y./ell).^2) .* exp( - sqrt(5) * pdist2(X./ell,Y./ell) ); % Matern 5/2
    k{4} = @(X,Y) max(zeros(size(X,1),size(Y,1)),1-pdist2(X./ell,Y./ell)).^3 .* (3*pdist2(X./ell,Y./ell) + 1); % C^2 Wendland (d=1)
    k{5} = @(X,Y) max(zeros(size(X,1),size(Y,1)),1-pdist2(X./ell,Y./ell)).^5 .* (8*pdist2(X./ell,Y./ell).^2 + 5*pdist2(X./ell,Y./ell) + 1); % C^4 Wendland (d=1)
    k{6} = @(X,Y) exp( - pdist2(X./ell,Y./ell).^2 ); % Gaussian
    
    % evaluation
    h_vals = 1./((1:10).^2);
    for j = 1:length(h_vals)
        h = h_vals(j);    
        Xh = h * Xn;
        fh = zeros(n,1);
        for i = 1:n
            fh(i) = f(Xh(i,:));
        end
    
        % classical extrapolation methods
        mn_Richardson = univariate_extrapolation(fh,Xh,"Richardson");
        mn_Shanks = univariate_extrapolation(fh,Xh,"Shanks");
        mn_Germain_Bonne = univariate_extrapolation(fh,Xh,"Germain-Bonne");
        mn_Thiele = univariate_extrapolation(fh,Xh,"Thiele");
        abs_error(1,j) = abs(mn_Richardson - f0);
        abs_error(2,j) = abs(mn_Shanks - f0);
        abs_error(3,j) = abs(mn_Germain_Bonne - f0);
        abs_error(4,j) = abs(mn_Thiele - f0);
    
        % Gauss--Richardson Extrapolation
        for m = 1:length(k)
            km = k{m}; % the kernel
            [mn,kn] = GP(fh,Xh,b,km);
            abs_error(4+m,j) = abs(mn(0) - f0);
            rel_error(m,j) = (f0 - mn(0)) / sqrt(kn(0,0));
        end
    
    end
    
    % plotting
    figure()
    set(gcf,'Position',[100,100,400,400])
    loglog(h_vals,abs_error(1,:),'k-o'); hold on;
    loglog(h_vals,abs_error(2,:),'k--o'); hold on;
    loglog(h_vals,abs_error(3,:),'k:o'); hold on;
    loglog(h_vals,abs_error(4,:),'k-.o'); hold on;
    loglog(h_vals,abs_error(5,:),'-^','Color',"#4DBEEE"); hold on;
    loglog(h_vals,abs_error(6,:),'--^','Color',"#4DBEEE"); hold on;
    loglog(h_vals,abs_error(7,:),':^','Color',"#4DBEEE"); hold on;
    loglog(h_vals,abs_error(8,:),'--square','Color',"#A2142F"); hold on;
    loglog(h_vals,abs_error(9,:),':square','Color',"#A2142F"); hold on;
    loglog(h_vals,abs_error(10,:),'-pentagram','Color',"#EDB120"); hold on;
    xlabel('h')
    ylabel('absolute error')
    box on
    legend({'Richardson','Shanks','Germain-Bonne','Thiele', ...
            'GRE (Matern, $s = 0$)','GRE (Matern, $s = 1$)','GRE (Matern, $s = 2$)', ...
            'GRE (Wendland, $s = 1$)','GRE (Wendland, $s = 2$)','GRE (Gaussian, $s = \infty$)'}, ...
            'Location','southeast')
    print(['abs_error_',num2str(application,'%u')])
    
    figure()
    set(gcf,'Position',[100,100,400,400])
    semilogx(h_vals,rel_error(1,:),'-^','Color',"#4DBEEE"); hold on;
    semilogx(h_vals,rel_error(2,:),'--^','Color',"#4DBEEE")
    semilogx(h_vals,rel_error(3,:),':^','Color',"#4DBEEE")
    semilogx(h_vals,rel_error(4,:),'--square','Color',"#A2142F")
    semilogx(h_vals,rel_error(5,:),':square','Color',"#A2142F")
    semilogx(h_vals,rel_error(6,:),'-pentagram','Color',"#EDB120")
    ylims = ylim;
    shade_background_2([min(h_vals),max(h_vals)],ylim,@(x,y) normpdf(y))
    ylim(ylims)
    xlabel('h')
    ylabel('relative error')
    box on
    if application == 1
        legend_loc = 'northeast';
    elseif application == 2
        legend_loc = 'southwest';
    end
    legend({'GRE (Matern, $s = 0$)','GRE (Matern, $s = 1$)','GRE (Matern, $s = 2$)', ...
            'GRE (Wendland, $s = 1$)','GRE (Wendland, $s = 2$)','GRE (Gaussian, $s = \infty$)'}, ...
            'Location',legend_loc)
    print(['rel_error_',num2str(application,'%u')])

end

