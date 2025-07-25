% modelweightsupdatesim Update of parameters for a Dirichlet prior on model weights
% USAGE
%   ap=modelweightsupdatesim(p,a);
% INPUTS
%   p   : q x n matrix of model probabilities
%   a   : q x n matrix of current values of belief distribution parameters
% OUTPUT
%   ap  : q x n matrix of updated belief distribution parameters
%
% The actual posterior is a mixture of Dirichlet distibutions with mixing
%      weights p*a/sum(p*a). Each column of ap  provides the parameters that 
%      match E[ln(w)] for the exact and approximate posterior distributions
%      associated with the corresponding columns of p and a.
%
% Calls fast MEX file modelweightsupdatec if available
function ap=modelweightsupdatesim(p,a,callmex)
[q,n] = size(a);
if size(p,1)~=q 
  error('a and p are not compatible')
end
if n==1
  n=size(p,2);
elseif (size(p,2)>1 && size(p,2)~=n)
  error('a and p are not compatible')
end
if nargin<3
  callmex=true;
end
eta=sum(a,1);
%ess = bsxfun(@minus,psi(a),psi(eta+1));
%g=bsxfun(@rdivide,p,sum(p.*a,1));
%ess=ess+g;
%a0=(1+g).*a;
g = p./sum(p.*a,1);
ess = psi(a)+g-psi(eta+1);
a0=(1+g).*a;
%a0=bsxfun(@rdivide,bsxfun(@plus,eta,c),eta+1).*a;
  
if callmex
  try
    ap=dirsolvec(ess,a0);
    if 0
      r = abs(psi(ap)-ess-psi(sum(ap)));
      ind = find(any(r.*ess>=1e-14,1));
      if ~isempty(ind)
        ap(:,ind) = dirsolve(ess(:,ind),a0(:,ind));
      end
    end
  catch
    ap=dirsolve(ess,a0);
  end
else
  ap=dirsolve(ess,a0);
end
return
    %r = psi(ap)-ess-psi(sum(ap,1));
 % disp(['# of convergence failures in m-file: ' num2str(sum(any(abs(r.*max(1,ess))>=1e-14,1)))]) 
    %ind = find(any(abs(r.*max(1,ess))>=1e-14,1));
    %disp(length(ind)) 

% check for "bad" updates
infind=any(isinf(ap) | isnan(ap),1);
if any(infind)
  if size(p,2)==1
     ap(:,infind) = a(:,infind) + p;
  else
  ap(:,infind) = a(:,infind) + p(:,infind);
  end
  ap(:,infind) = ap(:,infind) .* (eta(infind)+1)./eta(infind);
  fprintf('number of "bad results": %1.0f\n',sum(infind));
end

return

% if missing then no change
%ap(isnan(ap))=a(isnan(ap));
% otherwise eta must be at least as great as mineta
sap=sum(ap,1);
mineta = 1/size(ap,1);
ii = sap<mineta;
if any(ii)
  ap(:,ii) = bsxfun(@rdivide,ap(:,ii)*mineta,sap(ii)); 
end

return

if callmex
  try
    ap=modelweightsupdatec(p,a,ess,1);
  catch
    callmex=false;
  end
end
if ~callmex
  %g=bsxfun(@rdivide,p,sum(p.*a,1));
  g=p./sum(p.*a,1);
  ess=ess+g;
  a0=(1+g).*a;
  %a0=bsxfun(@rdivide,bsxfun(@plus,eta,c),eta+1).*a;
  ap=dirsolve(ess,a0);
end
return

% dirsolve Converts expected sufficient statistics into parameter values
% ess : expected sufficient statistics
% a   : initial guess of a
% Uses Newton's method
function a=dirsolve(ess,a)
tol=5e-13;
maxit=200;
vec=@(x) x(:);
for it=1:maxit
  eta=sum(a,1);
  %r=bsxfun(@minus,psi(a),psi(eta))-ess;   % residual function
  %r=psi(a)-psi(eta)-ess;   % residual function
  r=psi(a)-ess-psi(eta);   % residual function
  r(a<=0)=0;
  if all(vec(abs(r.*max(1,ess))) < tol)       % convergence check
    break;
  end
  % update a  
  pa1=1./psi(1,a);
  %da = bsxfun(@plus,r,sum(r.*pa1,1)./(1./psi(1,eta)-sum(pa1,1))).*pa1;
  da = (r + sum(r.*pa1,1)./(1./psi(1,eta)-sum(pa1,1))).*pa1;
  %da(a==0)=0;
  a=a-da;
  a(a<0)=1e-30;          % avoid negative values
end
if it>=maxit
  disp(['# of convergence failures in m-file: ' num2str(sum(any(abs(r.*max(1,ess))>=tol,1)))]) 
  
  %ind=find(any(abs(r.*max(1,ess))>=tol,1));
  %disp(max(sum(a(:,ind),1)))
  %disp(it)
  %warning('failure to converge in dirsolve')
end
 
