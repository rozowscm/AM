% Demonstrates the updating of weights on alternative models
% An n-category random variable Y has probability P*w
% where P is a known n x q probability matrix w/ non-negative values 
% and columns that sum to 1 and w is an unknown probability vector with a
% Dirichlet(a) prior. When event i occurs the vector p=P(i,:) provides the
% likelihood used to update the Dirichlet parameters to obtain an
% approximate posterior.

% Uses mdpsolve functions simplexgrid, normalizeP and randdisc
% mdpsolve is available at https://github.com/PaulFackler/MDPSolve


%
%Two Models
% initial setup
%%
clear
clc
n=3;
q=2;
T=100;
rep=100000;
w=[.65;.35];

P=[.2 .5;
   .3 .3;
   .5 .2];

p=P*w;
pdirichlet=@(x,a)exp(gammaln(sum(a))-sum(gammaln(a))+log(x+1e-30)*(a(:)-1)); 
x=simplexgrid(q,1000,1);
set(0,'DefaultTextInterpreter','LaTex','DefaultTextFontSize',16,'DefaultAxesFontSize',16)

%% Figure 1

set(0,'DefaultTextInterpreter','LaTex','DefaultTextFontSize',16,'DefaultAxesFontSize',16)

%b=[0.95;0.05];
%eta=.5;
%a=b*eta;

eta=1;
a=[1;1]*(eta/2);
b=[1;1]/2;


figure(1); clf
set(gcf,'units','normalized','position',[0.100 0.146 0.854 0.654])
set(gcf,'units','normalized','position',[0.003 0.146 0.995 0.654])
% panel a
subplot(1,2,1)
y=1;
tlab = 'Prior and posterior distributions - period 1';
[a,b]=TwoModelFigures(x,a,b,P(y,:)',tlab);

% panel b
subplot(1,2,2)
y=3;
tlab = 'Prior and posterior distributions - period 2';
[a,b]=TwoModelFigures(x,a,b,P(y,:)',tlab);
%return





%% Figure 2

% set random number generator state to produce the same results each time
s = RandStream('mt19937ar','Seed',17);
RandStream.setGlobalStream(s);

A=zeros(q,T+1);
B=zeros(q,T+1);
a=ones(q,1)/q;
b=ones(q,1)/q;
A(:,1)=a;
B(:,1)=[.5;.5];
y=randdisc(P*w,rand(T,1));
for i=1:T
  a=modelweightsupdate(P(y(i),:),a',[],0);
  A(:,i+1)=a;
  b=normalizeP(P(y(i),:)'.*b);
  B(:,i+1)=b;
end
W=bsxfun(@rdivide,A,sum(A,1));

figure(2); clf
set(gcf,'units','normalized','position',[0.086 0.229 0.880 0.474],'Name','')
subplot(1,2,1)
plot(0:T,B(1,:),'-k',0:T,W(1,:),'--k',[0 T],[w(1) w(1)],':k','linewidth',1.5)
ylim([0 1])
xlabel('time')
h=legend('$$b_1$$','$$w_1$$','location','eastoutside');
set(h,'Interpreter','latex')
title('Single time path')

disp('Last period estimates of w: exact, belief, model, regression on counts')
disp([w B(:,end) W(:,end) P\(histcounts(y,n)'/length(y))])

disp('30th period estimates of w: exact, belief, model, regression on counts')
disp([w B(:,30) W(:,30) P\(histcounts(y,n)'/length(y))])

a=ones(q,rep)/q;
b=ones(q,rep)/q;
for i=1:T
   y=randdisc(P*w,rand(rep,1));
%    for j=1:rep
%      a(:,j)=modelweightsupdate(P(y(j),:),a(:,j)');
%    end
   a=modelweightsupdatesim(P(y,:)',a,0);
   W(:,i+1)=mean(bsxfun(@rdivide,a,sum(a,1)),2);
   b=normalizeP(P(y,:)'.*b);
   B(:,i+1)=mean(b,2);
end

subplot(1,2,2)
plot(0:T,B(1,:),'-k',0:T,W(1,:),'--k',[0 T],[w(1) w(1)],':k','linewidth',1.5)
ylim([0 1])
xlabel('time')
h=legend('E$$[b_1]$$','E$$[w_1]$$','location','eastoutside');
set(h,'Interpreter','latex')
title('Expected time path')

disp('Last period estimates of w: exact, belief, model, regression on counts')
disp([w B(:,end) W(:,end) P\(histcounts(y,n)'/length(y))])

disp('30th period estimates of w: exact, belief, model, regression on counts')
disp([w B(:,30) W(:,30) P\(histcounts(y,n)'/length(y))])


%{


%% Figure 3
P=[exp(n:-1:1)',(n-(n/2))^2-((0:n-1)'-(n-1)/2).^2,exp(1:n)'];
P=normalizeP(P);

w=[3;1;6];
w=normalizeP(w);
p=P*w;            % true model wieghts
g=p'*log(P);      % gamma values for each model
disp(g)

figure(3); clf
u=(1:6)';

ww=[[1;1;1]/3 [1;0;1]/2 [1; 1; 0]/2]; 
plot(u,P*ww(:,1),'ks--',u,P*ww(:,2),'ko--',u,P*ww(:,3),'k^--','linewidth',1);
hold on; 
cc=0.9;
patch([1;u;flipud(u)],[P(1,3);P(:,1);flipud(P(:,3))],[cc cc cc])
patch([1;u;flipud(u)],[P(1,3);P(:,2);flipud(P(:,3))],[cc cc cc])

plot([1 1],[P(1,3) P(1,1)],'linewidth',1,'color',[cc cc cc]);
plot([6 6],[P(end,3) P(end,1)],'linewidth',1,'color',[cc cc cc]);
plot(u,P*ww(:,1),'ks--',u,P*ww(:,2),'ko--',u,P*ww(:,3),'k^--','linewidth',1);
plot(u,P,'k-*','linewidth',1);  
legend('w=[1/3  1/3  1/3] ','w=[1/2    0   1/2] ','w=[1/2  1/2    0 ] ','location','north')
hold off
text(1.2,0.6,'model 1','horizontalalignment','left')
text(3.5,0.27,'model 2','horizontalalignment','center')

text(5.8,0.6,'model 3','horizontalalignment','right')
xlim([0.75 6.25])
set(gcf,'name','Alternate models and example weighted average models')
xlabel('category','Color','k')

ylabel('probability')

%%
% Figures 4 and 5
T=30;
eta=1.5;
threemodelfigures(T,eta,4)
%%

%}

%{
% Figure 6
q=2;

pdirichlet=@(x,a)exp(gammaln(sum(a))-sum(gammaln(a))+log(x+1e-30)*(a(:)-1)); 

x=simplexgrid(q,1000,1);
w=[0.8;0.4];

w=[0.65;0.35];
P=[0.1 0.8; 0.5 0.15; 0.4 0.05];
p=P*w;

set(0,'DefaultTextInterpreter','LaTex','DefaultTextFontSize',16,'DefaultAxesFontSize',16)
%%
figure(6); clf
set(gcf,'units','normalized','position',[0.1 0.2 0.7 0.6]) 
eta=[1/w(1)/2 1/w(1) 1/w(2) 5 10];
m=length(eta);
lnames=cell(1,m);
for i=1:m
 z0=pdirichlet(x,w*eta(i));
 plot(x(:,1),z0,'k','linewidth',1+i)
 hold on
 lnames{i}=['$$~\alpha = [' num2str(w(1)*eta(i)) ',' num2str(w(2)*eta(i)) ']$$'];
end
xlabel('$$w_1$$','interpreter','latex')
ylim([0 3])
legend(lnames,'location','eastoutside','interpreter','latex')


%%
% Figure 7
eta=1.5;
a=[1;1]*(eta/2);
b=[1;1]/2;

figure(7); clf
set(gcf,'units','normalized','position',[0.100 0.146 0.854 0.654])
% panel a
subplot(1,2,1)
y=1;
tlab = 'Prior and posterior distributions - period 1';
[a,b]=TwoModelFigures(x,a,b,P(y,:)',tlab);

% panel b
subplot(1,2,2)
y=3;
tlab = 'Prior and posterior distributions - period 2';
[a,b]=TwoModelFigures(x,a,b,P(y,:)',tlab);
%%
% Figures 8 and 9
T=50;
eta=3/2;   % Jeffrey's prior for the Dirichlet-multinomial 
threemodelfigures(T,eta,8)

%return


%% 
%Figure 10 
figure(10); clf
set(gcf,'units','normalized','position',[0.1 0.2 0.7 0.6]) 
eta=[0.5 2 4];
m=length(eta);
w=[0.5;0.5];
lnames=cell(1,m);
for i=1:m
 z0=pdirichlet(x,w*eta(i));
 plot(x(:,1),z0,'k','linewidth',1+2*i)
 hold on
 lnames{i}=['$$~\eta = ' num2str(eta(i)) ' $$'];
end
xlabel('$$w_1$$','interpreter','latex')
ylim([0 3])
legend(lnames,'location','eastoutside','interpreter','latex')
%}

%%
%{
%Figure 11
% Eta comparisons
% set random number generator state to produce the same results each time
figure(11); clf
s = RandStream('mt19937ar','Seed',17);
RandStream.setGlobalStream(s);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
etas=[0.125 0.25 0.5 1 2 4 8];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=100;
rep=100000;
V(1,:)=[0.5 0.5 0.5 0.5 0.5]';
m=length(etas);
W=cell(1,m);

for j=1:m
W{j}=zeros(q,T+1);
W{j}(:,1)=[0.5 0.5];
eta=etas(j);
a=ones(q,rep)*(eta/2);
for i=1:T
   y=randdisc(P*w,rand(rep,1));
   a=modelweightsupdatesim(P(y,:)',a,0);
   v=bsxfun(@rdivide,a(:,1:5),sum(a(:,1:5),1));
   V(i+1,:)=v(1,:)';
   W{j}(:,i+1)=mean(bsxfun(@rdivide,a,sum(a,1)),2);
   %disp('Last period estimates of w: exact, belief, model, regression on counts')
   %disp([w W{j}(:,end) P\(histcounts(y,n)'/length(y))])
end
end
%%
for l=1:m
 subplot(1,m,l)
 plot(0:T,V(:,1)',0:T,V(:,2)',0:T,V(:,3)',0:T,V(:,4)',0:T,V(:,5)','--k','linewidth',1,'Color',[0 0 0]+0.05)
hold on
plot(0:T,W{l}(1,:),'-k','linewidth',3.5,'Color',l*[0.125 0.125 0.125])
ylim([0.3 1])
end
plot([0 T],[w(1) w(1)],':k','linewidth',3.5,'Color',[0 0 0])
xlabel('time')
ylabel('$$w_1$$','interpreter','latex')
title(['$$~\eta_0 $$'])
legend('0.125','0.25','0.5','1','2','4','8','location','north','FontSize',14);
%}

%{
%%
%Figure 12
% Model comparisons
% set random number generator state to produce the same results each time
s = RandStream('mt19937ar','Seed',17);
RandStream.setGlobalStream(s);
q=2;
T=100000;
W=zeros(q,T+1);
V(1,:)=[0.5 0.5 0.5 0.5 0.5]';
W(:,1)=[0.5 0.5];
rep=1000;

P=cell(1,3);
P{1}=[.65 .35;
      .35 .65];
P{2}=[.75 .30;
      .30 .75];
P{3}=[.95 .20;
      .20 .95];
eta = 1;
w=[1/3;2/3];

for j=1:3
p=P{j};
a=ones(q,rep)*(eta/2);
for i=1:T
   y=randdisc(p*w,rand(rep,1));
   a=modelweightsupdatesim(p(y,:)',a,0);
   v=bsxfun(@rdivide,a(:,1:5),sum(a(:,1:5),1));
   V(i+1,:)=v(1,:)';
   W(:,i+1)=mean(bsxfun(@rdivide,a,sum(a,1)),2);
end

lnames{j}=['P = [' num2str(p(1)) ',' num2str(p(2)) ']'];
subplot(1,3,j)
plot(0:T,V(:,1)',0:T,V(:,2)',0:T,V(:,3)',0:T,V(:,4)',0:T,V(:,5)','--k','linewidth',1,'Color',[0 0 0]+0.05)
hold on
plot(0:T,W(1,:),'-k',[0 T],[w(1) w(1)],':k','linewidth',3.5,'Color',[0 0 0])
ylim([0 1])
xlabel('time')
ylabel('$$w_1$$','interpreter','latex')
title(lnames{j})

disp('Last period estimates of w: exact, belief, model, regression on counts')
disp([w W(:,end)])

%}



%{
% Figures 13 and 14
T=50;
eta=3;   % Uniform prior for the Dirichlet-multinomial 
threemodelfigures(T,eta,13)


%}


%%
