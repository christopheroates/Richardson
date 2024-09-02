%% Explore optimal design mathodology in dimension d = 1

clear all

% design problem set up
d = 1;
b = @(X) X; % error bound
c = @(X) X.^(-1); % cost

% higher precision arithmetic
digits(100)
ell = vpa(1); % make length-scale higher precision to force all to be higher precision

% pick two kernels to compare designs
k{1} = @(X,Y) exp( - pdist2(X./ell,Y./ell) ); % Matern 1/2
k{2} = @(X,Y) exp( - pdist2(X./ell,Y./ell).^2 ); % Gaussian

% computation budget
C_vals = 10:5:50;
X = cell(length(C_vals),2);

% candidate experiments
m = 20;
xi_vals = 0.8.^(1:m)'; % univariate grid
min_num_points = 1;
max_num_points = 20;
constraint_type = 'sum';

% compute optimal designs
for i = 1:length(C_vals)
    C = C_vals(i);
    disp(C)
    for j = 1:2
        [X{i,j},cost] = design(k{j},b,c,C,xi_vals,min_num_points,max_num_points,constraint_type); 
    end
end

% plotting
figure(1)
set(gcf,'Position',[100,100,400,400])
for i = 1:length(C_vals)
    C = C_vals(i);
    semilogx([0.000000001,1],[C,C],'k:'); hold on;
    plt1 = plot(X{i,1},C * ones(length(X{i,1}),1),'^','Color',"#4DBEEE",'MarkerEdgeColor',"#4DBEEE");
    plt2 = plot(X{i,2},C * ones(length(X{i,2}),1),'pentagram','Color',"#EDB120",'MarkerEdgeColor',"#EDB120");
end
xlim([0.01,1])
ylims = [min(C_vals)-3,max(C_vals)+3];
ylim(ylims)
for i = 1:length(xi_vals)
    plot([xi_vals(i),xi_vals(i)],ylims,'k:')
end
xlabel('$x$')
ylabel('$C$')
legend([plt1(1),plt2(1)],'Matern ($s = 0$)','Gaussian ($s = \infty$)','Location','southwest')
print('design_1')