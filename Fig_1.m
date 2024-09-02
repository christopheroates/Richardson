%% Reproduces illustration from the main text
% Based on finite difference approximation.

clear all

% simulated data
g = @(T) 1 + sin(1.5 * pi * T).^2;
t0 = 0;
f = @(X) (g(t0+X) - g(t0)) ./ X; % finite difference approximation to g'(t0)
f0 = 0;
n = 5;
Xn = 0.01 + 1./(2:(n+1))';
fn = f(Xn);

r = 1; % order of bound
b = @(X) X; % error bound

% cross validation for kernel parameters
ell_vals = vpa([0.01,0.05,0.1,0.2,0.5,1,1.5,2,5,10]); % candidate length scale parameter values
digits(100) % higher precision arithmetic
wb = waitbar(0,'Model selection');
for i = 1:length(ell_vals)

    waitbar((i-1)/length(ell_vals),wb)
    
    % kernel
    ell = ell_vals(i);
    k{i} = @(X,Y) (1 + sqrt(5) * pdist2(X./ell,Y./ell) + (5/3) * pdist2(X./ell,Y./ell).^2) .* exp( - sqrt(5) * pdist2(X./ell,Y./ell) ); % Matern 5/2
        
    % negative (2x) log quasi likelihood
    [~,~,nlql(i)] = GP(fn,Xn,b,k{i});

end

close(wb)

% final model
[~,ix] = min(nlql); 
ell = ell_vals(ix);
k = k{ix};
[mn,kn] = GP(fn,Xn,b,k);

% plotting
figure(1)
set(gcf,'Position',[100,100,800,400])
clf
n_plot = 200;
X_plot = linspace(0,1,n_plot)';
f_vals = f(X_plot);
mn_vals = double(mn(X_plot));
std_vals = zeros(n_plot,1);
for i = 1:length(X_plot)
    std_vals(i) = double(kn(X_plot(i),X_plot(i)).^(1/2));
end
plot(X_plot,f_vals,'b-'); hold on;
plot(0,f0,'bpentagram')
plot(Xn,fn,'ro')
plot(X_plot,mn_vals,'k--'); 
plot(X_plot,mn_vals + std_vals,'k:')
plot(X_plot,mn_vals - std_vals,'k:')
xlabel('$x$')
ylabel('$f(x)$')
legend({'$f(x)$','$f(0)$','$f(x_n)$','$m_n[f](x)$','$\sqrt{k_n[f](x,x)}$'}, ...
       'Location','southwest')
print('illustration')