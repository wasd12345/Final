function [x] = IRLS(A,b,p,x0,nIter)
%Based on the algorithm outlined in:
%Iterative Reweighted Least Squares
% C. Sidney Burrus
% OpenStax-CNX

%A, x, b are standard matrix,vectors of least squares notation
%p is the norm (i.e. here used 1 for L1 norm)
%nIter is number of iterations

'Solving with IRLS...'

%Should already be sparse, but check just in case:
%issparse(A)
%issparse(b)
%issparse(x0)


%Use initial estimate that we worked hard to get by using Edge Flow
x  = x0;

M = length(b);
N = length(x);

for k=1:nIter
    
    %Print iteration number
    'IRLS iteration'
    k
    
    %W is MxM, A is MxN, b is Mx1, x is Nx1, REGULARIZATION is NxN
    
    
    e  = A*x - b;                      % Residual
    w  = abs(e).^((p-2)/2);            % Error weights
    
    %Make sparse diagonal weights matrix W
    i = (1:M);
    W = sparse(i,i,w,M,M);
    WA = W*A; %is MxN matrix
    
    %If doing an increasing penalty term instead of just fixed cost:
    %Very briefly tried an exponential penalty schedule where lambda is
    %ramped up on each iteration. Decided not to play around with
    %hyperparameter tuning, but could potentially be helpful. See paper
    %referenced above for discussion of penalty ramp up/down, as well as
    %similar idea for the value of p (e.g. start at p=2 for L2 norm, then
    %decrease in steps until p=1).
    %lambda = 10^nIter;                   %Increase lambda exponentially to increase regularization penalty
    
    %Just used fixed large value for penalty:
    %e.g. %Xue et al. used a constant penalty of 1e5
    lambda = 100000. %10000
    
    %Determine which variables are out of bounds and need to be penalized:
    lb = sparse((min(0.,x)).^2);
    ub = sparse((min(0.,1.-x)).^2);
    
    %Make the N x N regularization diagonal matrix that tells how much each
    %element of x should be penalized based on how far out of bounds it is
    i = (1:N);
    REGULARIZATION = sparse(i,i,lambda*(lb+ub),N,N);
    
    %Solve this iteration for x.
    %Even though the terms are sparse, takes very long time...
    x  = (WA'*WA + REGULARIZATION)\(WA'*W)*b;

end

end