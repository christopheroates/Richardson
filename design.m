%% Experimental design for Gauss--Richardson Extrapolation
%
% Inputs:
% ke              = function handle ke(x,y) to kernel
% b               = function handle b(x) to error bound
% c               = function handle c(x) to computational cost
% C               = computational budget
% XN              = (N x d) array of candidate values for discretisation parameters x_i in R^d
% min_num_points  = minimum number of elements in returned design
% max_num_points  = maximum number of elements in returned design
% constraint_type = 'sum' (sum of c(xi) used) or 'max' (max of c(xi) used)
%
% Outputs:
% X    = (n x d) array of selected values for discretisation parameters x_i in R^d
% cost = the cost of the experimental design X

function [X,cost] = design(ke,b,c,C,XN,min_num_points,max_num_points,constraint_type)

% candidate states
N = size(XN,1);
bN = b(XN);
KN = ke(XN,XN);
KbN = bN .* KN .* (bN');
cN = c(XN);

if strcmp(constraint_type,'max')
    idx = (cN < C);
    X = XN(idx,:);
    cost = sum(cN(idx));
end

if strcmp(constraint_type,'sum')

    % initialise optimal states and cost
    X = []; % optimal states
    val = 0; % objective function value
    cost = inf; % cost of optimal states
    
    % handle max_num_points = inf
    max_num_points = min(max_num_points,N);
    
    wb = waitbar(0,'Experimental Design');
    for n = min_num_points:max_num_points
    
        waitbar((n-min_num_points)/(max_num_points),wb);
        
        % candidate subsets of size n
        subsets = nchoosek(1:N,n);
    
        for i = 1:size(subsets,1)
            idx = subsets(i,:);
            new_cost = sum(cN(idx));
            if new_cost < C
                % check if design is saturated (if not, then it's suboptimal)
                if n < max_num_points
                    remaining_budget = C - new_cost;
                    cheapest_extra_experiment = min(cN(setdiff(1:N,idx)));
                    saturated = (remaining_budget < cheapest_extra_experiment);
                else
                    saturated = true;
                end
                if saturated
                    % use double precision to see if worth investigating at
                    % higher precision
                    approx_new_val = ones(1,n) * (double(KbN(idx,idx)) \ ones(n,1));
                    if approx_new_val > 0.9 * val
                        new_val = ones(1,n) * (KbN(idx,idx) \ ones(n,1));
                        if new_val > val
                            X = XN(idx,:);
                            cost = new_cost;
                            val = new_val;
                        end
                    end
                end
            end
        end
    
    end

    close(wb)

end

end

