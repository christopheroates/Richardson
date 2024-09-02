function PR = perms_rep(N,K)
% This is (basically) the same as npermutek found on the FEX.  It is the  
% fastest way to calculate these (in MATLAB) that I know.  
% pr = @(N,K) N^K;  Number of rows.
% A speed comparison could be made with COMBN.m, found on the FEX.  This
% is an excellent code which uses ndgrid.  COMBN is written by Jos.
%
%      % All timings represent the best of 4 consecutive runs.
%      % All timings shown in subfunction notes used this configuration:
%      % 2007a 64-bit, Intel Xeon, win xp 64, 16 GB RAM  
%      tic,Tc = combinator(single(9),7,'p','r');toc  
%      %Elapsed time is 0.199397 seconds.  Allow Ctrl+T+C+R on block
%      tic,Tj = combn(single(1:9),7);toc  
%      %Elapsed time is 0.934780 seconds.
%      isequal(Tc,Tj)  % Yes
if N==1
   PR = ones(1,K,class(N)); 
   return
elseif K==1
    PR = (1:N).';
    return
end
CN = class(N);
M = double(N);  % Single will give us trouble on indexing.
L = M^K;  % This is the number of rows the outputs will have.
PR = zeros(L,K,CN);  % Preallocation.
D = ones(1,N-1,CN);  % Use this for cumsumming later.
LD = M-1;  % See comment on N. 
VL = [-(N-1) D].';  % These values will be put into PR.
% Now start building the matrix.
TMP = VL(:,ones(L/M,1,CN));  % Instead of repmatting.
PR(:,K) = TMP(:);  % We don't need to do two these in loop.
PR(1:M^(K-1):L,1) = VL;  % The first column is the simplest.
% Here we have to build the cols of PR the rest of the way.
for ii = K-1:-1:2
    ROWS = 1:M^(ii-1):L;  % Indices into the rows for this col.
    TMP = VL(:,ones(length(ROWS)/(LD+1),1,CN));  % Match dimension.
    PR(ROWS,K-ii+1) = TMP(:);  % Build it up, insert values.
end
PR(1,:) = 1;  % For proper cumsumming.
PR = cumsum2(PR);  % This is the time hog.

end

function A = cumsum2(A)
%CUMSUM2, works with integer classes. 
% Duplicates the action of cumsum, but for integer classes.
% If Matlab ever allows cumsum to work for integer classes, we can remove 
% this.
if isfloat(A)
    A = cumsum(A);  % For single and double, use built-in.
    return
else
    try
        A = cumsumall(A);  % User has the MEX-File ready?
    catch
        warning('Cumsumming by loop.  MEX cumsumall.cpp for speed.') 
        for ii = 2:size(A,1)
            A(ii,:) = A(ii,:) + A(ii-1,:); % User likes it slow.
        end
    end
end

end


