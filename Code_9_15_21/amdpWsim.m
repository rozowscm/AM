% amdpWsim Simulates a weighted Adaptive Management model
% INPUTS
%   p      : q element cell array containing a set of n x m probability matrices
%   wstar  : a q-vector of weights (summming to 1) representing the true model
%   T      : # of time periods for simulation
%   S0     : reps x 1 vector of initial states (each on {1,...,n})
%   Alpha0 : reps x q (or 1 x q) matrix of initial belief parameters
%   Afunc  : function to determine actions with form: A = Afunc(S,Alpha)
%              where S and Alpha are reps x 1 and reps x q respectively
% OUTPUTS
%   S : reps x T matrix of simulated state values
%   A : reps x T matrix of simulated action values
%   Alpha : reps x q x T array of simulated belief parameter values
%
function [S,A,Alpha,Sb,Ab,B]=amdpWsim(p,wstar,T,St,Alphat,bt,Afunc,MM)
if MM
    alg=@cds96;
else
    alg=@modelweightsupdatesim;
end
q=length(p);        %# of models
[n,m]=size(p{1});   % number of states and state/action conbinations
Sbt=St;
% compute true transition probability matrix
pstar = zeros(n,m);
for i=1:q
  pstar = pstar + wstar(i)*p{i};
end
reps=length(St); % # of replications
S=zeros(reps,T); Sb=zeros(reps,T);% states
Alpha=zeros(reps,q,T); B=Alpha;  % belief parameters
if size(Alphat,1)==1
  Alphat=repmat(Alphat,reps,1);
end
if size(bt,1)==1
  bt=repmat(bt,reps,1);
end
A=zeros(reps,T); Ab=A;  % actions
lk=zeros(q,reps); lkb=lk;
% loop over time periods
for t=1:T
  B(:,:,t)=bt;  
  At = Afunc(St,Alphat); bt = Afunc(Sbt,bt);  % current action
  Xt=St+n*(At-1); Vt=Sbt+n*(bt-1);   % current state/action
  % store current values
  S(:,t)=St; Sb(:,t)=Sbt;
  Alpha(:,:,t)=Alphat;
  A(:,t)=At; Ab(:,t)=bt;
  % next period state
  St = randdisc(pstar,rand(reps,1),Xt); Sbt = randdisc(pstar,rand(reps,1),Vt); 
  % next period belief
  for i=1:q
    lk(i,:)=p{i}(St+n*(Xt-1));  
    lkb(i,:)=p{i}(Sbt+n*(Xt-1));
  end
  bt=lkb'.*B(:,:,t);
  bt=normalizeP(bt')';  
  Alphat=alg(lk,Alphat')';
end
  
  
