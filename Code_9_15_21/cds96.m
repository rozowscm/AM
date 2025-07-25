% cds96 Uses the mixture weights update rule from Section 3.2 Cowell, Dawid and Sebastiani (1996) 
% USAGE
%   ap=cds96(p,a);
% INPUTS
%   p        : q x n matrix of model probabilities
%   a        : q x n matrix of current values of belief distribution parameters
%   weighted : 0/1, 0 to use simple average of variance, 
%                   1 for weighted average, weighted by p.*a
% OUTPUT
%   ap  : q x n matrix of updated belief distribution parameters
%
% Robert G. Cowell and A. Philip Dawid and Paola Sebastiani. 1996. 
% A comparison of sequential learning methods for incomplete data. 
% Bayesian Statistics 5
% J. M. Bernardo, J. O. Berger, A. P. Dawid, and A. F. M. Smith, eds. 
% Oxford: University Press. 533-541.
function ap=cds96(p,a,alg)
if nargin<3, alg=0; end
%p=normalizeP(p);
pi=normalizeP(p.*a);
eta=sum(a,1);
m1=(a+pi)./(eta+1);
m2=(a+1).*(a+2*pi)./(eta+1)./(eta+2);
switch alg
  case 0
    beta = sum(m1-m2,1)./sum(m2-m1.^2,1); 
  case 1
    v=m2-m1.^2;
    [v,ii]=max(v,[],1);
    beta=m1-m2;
    [d,n]=size(beta);
    beta=beta((0:d:((n-1)*d)) + ii)./v;
  case 2
    beta = sum(pi.*(m1-m2),1)./sum(pi.*(m2-m1.^2),1);
    %[v,ii]=max(m2-m1.^2,[],1);
    %beta=m1.*(1-m1);
    %beta=beta(ii+(0:size(m1,1):numel(m1)-1))./v-1;
end
z=(eta-beta).*(sum((a+pi).^2)-(eta+1).^2.*sum(pi.^2));
z=sum(m1.^2)-sum(pi.^2);
if ~all(z.*(eta-beta)>=-eps)
  %disp(' ')
end
ii=sum(m1-m2)==0;
if length(eta)>1
  beta(ii)=eta(ii)+1;
else
  beta(ii)=eta+1;
end
ap=beta.*m1;

if any(eta+1-beta>1e12)
  disp(' ')
end

[q,n]=size(ap); z=ap-a; [~,ind0]=min(p); [~,ind1]=max(p); 
if max(z(ind0+(0:q:q*n-1)))>0 || min(z(ind1+(0:q:q*n-1)))<0
  %disp(' ')
end
return
