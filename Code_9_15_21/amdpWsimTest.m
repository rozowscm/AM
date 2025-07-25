q=3;
n=2;
m=8;
reps=1000;
T = 100;
p=cell(1,q);
for i=1:q
  p{i} = normalizeP(rand(n,m));
end
wstar=normalizeP(rand(q,1));
S0 = repmat(2,reps,1);
Alpha0=[1 1 1]/2; 
Afunc = @(S,Alpha) ones(size(S,1),1);
[S,A,Alpha]=amdpWsim(p,wstar,T,S0,Alpha0,Afunc);

mean(Alpha(:,:,end))
