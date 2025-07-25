% amdpdir Adaptive Management using Dirichlet priors
% USAGE
%   [g1,g2,Pa,Ra,Sa,Xa,Ixa]=amdpdir(inc,P,R,S,X,Ix,M,MM);
% INPUTS
%   inc     : positive integer, define the grid increment to be 1/inc
%   P       : ns x nx column stochastic transition matrix
%   R       : reward vector
%   S       : state matrix for no model uncertainty case
%   X       : state/action matrix for no model uncertainty case
%   Ix      : state index for no model uncertainty case
%   M       : maximal value for sum(alpha_i)
%   MM      : method of moments? MM=1 for CDS96 MM=0 for KL divergence
% OUTPUTS
%   alpha    : matrix of alpha grid values
%   Pa       : transition matrix for the augmented problem
%   Sa       : state matrix for the augmented problem
%   Xa       : state/action matrix for the augmented problem
%   Ixa      : state index for the augmented problem
%
% Set up an adaptive management problem with one or more columns of p containing 
% uncertain values that are represented by a Dirichlet distribution. Each of the
% uncertain parameter values are represented by a value of alpha. Use
%   Ep=amdpdirp(p,alpha);
% to obtain the expected value of the p matrix for a specified alpha vector
function [w_eta,Pw,Ra,Sa,Xa,Ixa]=amdpdirichlet(p,P,R,S,X,Ix,maxEta,MM)
%%
%this section copied and adapted from ampd.m 5/27/21-cmr
if ~exist('R','var'),     R    =[]; end
if ~exist('S','var'),     S    =[]; end
if ~exist('X','var'),     X    =[]; end
if ~exist('Ix','var'),    Ix   =[]; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine number of models and ensure that R and P are compatible
% If both R and P are scalar the model is not adaptive
if isnumeric(P)
  P={P};
end
if iscell(P)
  qp=numel(P);           % number of alternative models for P
  if qp==1, probtype=0;
  else      probtype=1;
  end
else
  error('Incorrect P type')
end

if isnumeric(R)
  R={R};
end
if iscell(R)
  qr=numel(R);           % number of alternative models for R
  if qr>1, probtype=probtype+2; end
else
  error('Incorrect R type')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch probtype
  case 0              % P and R are both singleton
    Pa=P{1};
    Ra=R{1};
  case 1              % R is singleton
    q=qp;
    R=repmat(R,q,1); 
  case 2              % P is singleton
    q=qr;
    P=repmat(P,q,1);
  case 3              % P and R are both model dependent
    if qr~=qp
      error('R and P are incompatible')
    end
    q=qr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem is not adaptive
% both R and P are matrices or contain only 1 element
if probtype==0
  if nargout>=4
    if nargin>=4
      Sa=S;
    else
      Sa=(1:size(R,1))';
    end
  end
  warning('AMDP:ProblemNotAdaptive','Problem is not adaptive')
  return 
end

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ns,nx]=size(P{1});
if nx<ns
  warning('AMDP:TransposedOnly','AMDP implemented for transposed P matrices only!!')
end
warning('off','SIMPLEXBAS:Extrapolation')



%%%%%%%%%%%%%% non-zero weights/alphas allowed for now %%%%%%%%%%%%%%%%%%%%%%
w=simplexgrid(q,p,1); %weight grid
minEta=1/10; %minimum value for sum(alpha_i)= 1/#models is a guide
if numel(maxEta)==1
  etas=linspace(minEta,maxEta,p+1)'; %eta grid 
else
  etas=maxEta(:);
end
w_eta=rectgrid(w,etas);
N=size(w_eta,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ra=kron(R{1}(:),w_eta(:,1));
  for i=2:q
    Ra=Ra+kron(R{i}(:),w_eta(:,i));
  end

Sa=[kron(speye(ns),ones(N,1))*S repmat(w_eta,ns,1)];

Xa=[kron(speye(nx),ones(N,1))*X kron(ones(nx,1),speye(N))*w_eta];

Ixa=ones(N,1)*((Ix(:)'-1)*N) + ((1:N)')*ones(1,nx);
Ixa=Ixa(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%current algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if MM
    Pw=getPw(q,p,ns,nx,P,w_eta,etas);
else
    Pw=getPw2(q,p,ns,nx,P,w_eta,etas);
end
return

function Pw=getPw(q,p,ns,nx,P,w_eta,eta)
w=w_eta(:,1:q);
alpha=(w.*w_eta(:,q+1))';
Pw=cell(ns,nx);
pij=zeros(q,1);
for i=1:ns
  for j=1:nx
    for k=1:q, pij(k)=P{k}(i,j); end
    alphaplus = modelweightsupdatesim(pij,alpha,1)';
    etaplus=sum(alphaplus,2);
    wplus=alphaplus./etaplus;
    Pw{i,j}=kroncol(simplexbas(wplus,q,p,1),mxv(rectbas(etaplus,eta),w*pij));
  end
end
Pw=cell2mat(Pw);



function Pw=getPw2(q,p,ns,nx,P,w_eta,eta)
w=w_eta(:,1:q);
alpha=(w.*w_eta(:,q+1))';
Pw=cell(ns,nx);
pij=zeros(q,1);
for i=1:ns
  for j=1:nx
    for k=1:q, pij(k)=P{k}(i,j); end
    alphaplus = cds96(pij,alpha)';
    etaplus=sum(alphaplus,2);
    wplus=alphaplus./etaplus;
    Pw{i,j}=kroncol(simplexbas(wplus,q,p,1),mxv(rectbas(etaplus,eta),w*pij));
  end
end
Pw=cell2mat(Pw);

% 
% function Pw=getPw2(q,p,ns,nx,P,w_eta,etas) 
% %%%%%%%%%%
% N = size(w_eta,1); % number of alpha values
% P = reshape(cat(2,P{:}),[],q)';
% wws=sum(kroncol(P',w_eta(:,1:q)),2)';
% %ind=find(wws); wws=wws(ind); wws=full(wws);
% alphas = (w_eta(:,1:q) .* w_eta(:,q+1))';
% np=size(P,2);
% ww=zeros(q,N,np);
% for j=1:np
%     ww(:,:,j)=modelweightsupdatesim(P(:,j),alphas,1);
% end
% ww=reshape(ww,q,N*np);
% eta=sum(ww,1);
% ww=ww./eta;
% Pw=simplexbas(ww',q,p,1);  % get interpolation values for weights
% clear ww
% Pn=rectbas(eta',etas,1,0);  % get interpolation values for etas
% clear eta
% Pn=mxv(Pn,wws); %weight according to the apriori outcome probability
% clear wws
% Pw=kroncol(Pw,Pn); %combine interpolation weights and etas
% clear Pn
% Pw=sppermute(Pw,[1 3 2 4],[N N ns nx],[N*ns N*nx]);
% 

% 
% function Pw=getPw(q,p,ns,nx,P,w_eta,eta)
% w=w_eta(:,1:q);
% alpha=(w.*w_eta(:,q+1))';
% N=size(alpha,2);
% Pa=cell(ns,nx);
% pij=zeros(q,1);
% for i=1:ns
%   for j=1:nx
%     for k=1:q, pij(k)=P{k}(i,j); end
%     alphaplus = modelweightsupdatesim(repmat(pij,1,N),alpha)';
%     etaplus=sum(alphaplus,2);
%     wplus=alphaplus./etaplus;
%     Pa{i,j}=kroncol(simplexbas(wplus,q,p,1),rectbas(etaplus,eta));
%     Pa{i,j}=Pa{i,j}.*(w*pij)';
%   end
% end
% Pw=cell2mat(Pa);






