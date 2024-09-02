%% Explore optimal design mathodology in dimension d = 3

clear all

% design problem set up
d = 3;
b = @(X) X(:,1).^2 + X(:,2).^2 + X(:,3).^2; % error bound
c = @(X) X(:,1).^(-1) .* X(:,2).^(-1) .* X(:,3).^(-1); % cost

% higher precision arithmetic
digits(100)
ell = [vpa(1),vpa(1),vpa(1)]; % make length-scale higher precision to force all to be higher precision

% pick two kernels to compare designs
k{1} = @(X,Y) exp( - pdist2(X./ell,Y./ell) ); % Matern 1/2
k{2} = @(X,Y) exp( - pdist2(X./ell,Y./ell).^2 ); % exponentiated quadratic

% computational budget
C = 50;

% candidate experiments
m = 5;
xi_vals = 1./(m:-1:1); % univariate grid
XN = xi_vals(perms_rep(m,d));
min_num_points = 1;
max_num_points = 4;
constraint_type = 'sum';

% compute optimal designs
for i = 1:2
    [X{i},cost] = design(k{i},b,c,C,XN,min_num_points,max_num_points,constraint_type); 
end

% plotting
figure(1)
set(gcf,'Position',[100,100,400,400])
plt1 = plot3(X{1}(:,1),X{1}(:,2),X{1}(:,3),'^','Color',"#4DBEEE",'MarkerEdgeColor',"#4DBEEE"); hold on;
plt2 = plot3(X{2}(:,1),X{2}(:,2),X{2}(:,3),'pentagram','Color',"#EDB120",'MarkerEdgeColor',"#EDB120"); 
for i = 1:2
    for j = 1:size(X{i},1)
        plot3([0,X{i}(j,1)],[X{i}(j,2),X{i}(j,2)],[X{i}(j,3),X{i}(j,3)],'k:')
        plot3([X{i}(j,1),X{i}(j,1)],[0,X{i}(j,2)],[X{i}(j,3),X{i}(j,3)],'k:')
        plot3([X{i}(j,1),X{i}(j,1)],[X{i}(j,2),X{i}(j,2)],[0,X{i}(j,3)],'k:')
        %plot3([0,0],[0,X{i}(j,2)],[X{i}(j,3),X{i}(j,3)],'k:')
        plot3([0,0],[X{i}(j,2),X{i}(j,2)],[0,X{i}(j,3)],'k:')
        %plot3([0,X{i}(j,1)],[0,0],[X{i}(j,3),X{i}(j,3)],'k:')
        plot3([X{i}(j,1),X{i}(j,1)],[0,0],[0,X{i}(j,3)],'k:')
        plot3([0,X{i}(j,1)],[X{i}(j,2),X{i}(j,2)],[0,0],'k:')
        plot3([X{i}(j,1),X{i}(j,1)],[0,X{i}(j,2)],[0,0],'k:')
    end
end
up_limits = max([X{1};X{2}],[],1);
xlim([0,up_limits(1)])
ylim([0,up_limits(2)])
zlim([0,up_limits(3)])
xlabel('$x_1$')
ylabel('$x_2$')
zlabel('$x_3$')
legend([plt1(1),plt2(1)],'Matern ($s = 0$)','Gaussian ($s = \infty$)','Location','northeast')
print('design_3')