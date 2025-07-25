
n=5;
q=3;

w=normalizeP(rand(q,1));
P=0;
Pk=cell(1,q);
pik=cell(1,q);
for i=1:q
  Pk{i}=normalizeP(rand(n,n));
  pik{i}=longrunP(Pk{i});
  P=P+Pk{i}*w(i);
end

% get LR distribution
pi=ones(n,1)/n; for i=1:100; pi=P*pi; end;

S=(1:n)';
D=[];
D=add2diagram(D,'S','s',1,{},S);
D=add2diagram(D,'S+','f',1,{'S'},rvdef('d',P,S));

%%

reps=1000;
T=50;

s0=randdisc(pi,rand(reps,1));
[Y]=dsim(D,s0,T,[],[],1);
b=zeros(reps,q,T+1);
b(:,:,1)=repmat(ones(1,q)/q,reps,1);
Pc=zeros(1,q);
for t=1:T
  for j=1:reps
    for k=1:q
      Pc(k)=Pk{k}(Y{2}(j,t),Y{1}(j,t));
      bjt=Pc.*squeeze(b(j,:,t));
      b(j,:,t+1)=bjt/sum(bjt);
    end
  end
end


gamma=zeros(1,q);
bb=squeeze(mean(b,1));
figure(4); clf
tt=(0:T);
cc='gbr';
for i=1:q
  plot(tt,bb(i,:),cc(i))
  hold on
  gamma(i) = sum(P.*log(Pk{i}),1)*pi;
end
hold off
xlabel('time')
ylabel('b')
ylim([0 1])
legend('1','2','3','location','northwest')


gamma
[~,ii]=max(gamma)
