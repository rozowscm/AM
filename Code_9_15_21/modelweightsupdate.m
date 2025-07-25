% modelweightsupdate Update of parameters for a Dirichlet prior on model weights
% USAGE
%   ap=modelweightsupdate(p,a,ess);
% INPUTS
%   p   : n x q matrix of model probabilities
%   a   : m x q matrix of current values of belief distribution parameters
%   ess : psi(a)-psi(sum(a,2)+1) [precompute if calling multiple times)
% OUTPUT
%   ap  : q x m x n matrix of updated belief distribution parameters
%
% The actual posterior is a mixture of Dirichlet distibutions with mixing
%      weights pa/sum(pa). The ap vector provides the parameters that match 
%      E[ln(w)] for the exact and approximate posterior distributions.
%
% Calls fast MEX file modelweightsupdatec if available
function ap=modelweightsupdate(p,a,ess,callmex)
[ns,q] = size(p);
nb = size(a,1);
if ~isequal(size(a),[nb,q])
  error('a is incorrect size')
end
if nargin<4
  callmex=true;
end
if nargin<3  || isempty(ess)
  eta=sum(a,2);
  %a=bsxfun(@rdivide,a,eta);
  ess = bsxfun(@minus,psi(a),psi(eta+1));
else
  if ~isequal(size(ess),[nb,q])
    error('ess is incorrect size')
  end
end

if callmex
  try
    ap=modelweightsupdatec(p,a,ess);
    ap = reshape(ap,[q,nb,ns]);
  catch
    warning('mex update failed')
    callmex=false;
  end
end
if ~callmex
  ap = zeros(q,nb,ns);
  for k = 1:ns
    pk = p(k,:);
    for j = 1:nb
      aj = a(j,:);
      %eta=sum(aj);
      spa=sum(pk.*aj);
      c=pk/spa;
      %a0=(eta+c)/(eta+1).*aj;
      a0=aj+c.*aj;
      ap(:,j,k) = dirsolve(ess(j,:)+c,a0);
    end
  end
end
return

% dirsolve Converts expected sufficient statistics into parameter values
% ess : expected sufficient statistics
% a   : initial guess of a
% Uses Newton's method
function a=dirsolve(ess,a)
tol=1e-14;
maxit=25000;
for it=1:maxit
  %ind = a<1e-8;
  eta=sum(a);
  r=psi(a)-psi(eta)-ess;   % residual function
  %r(ind)=0;
  if norm(r./ess,inf)<tol       % convergence check
    break;
  end
  % update a
  %pa1=1./trigamma(a);
  %da = (r + (sum(r.*pa1)/(1/trigamma(eta)-sum(pa1)))).*pa1;
  pa1=1./psi(1,a);
  da = (r + (sum(r.*pa1)/(1/psi(1,eta)-sum(pa1)))).*pa1;
  %da(ind)=0;
  while any(da>a), da=da/2; end
  a = a-da;
  a=max(1e-30,a);
  %a(ind)=1e-30;          % avoid negative values
end
%disp(it)
if it>=maxit
  warning('failure to converge in dirsolve')
  disp(r')
end

 