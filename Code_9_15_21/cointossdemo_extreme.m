%implements the coin toss experiment in:
%Ziheng Yang, Tianqi Zhu. Overconfident Bayesian model selection.
%Proceedings of the National Academy of Sciences Feb 2018, 201712673; 
%DOI: 10.1073/pnas.1712673115

% Suppose the model set includes 2 models, one that a coin has probability
% of heads of p1, the other of p2. Suppose that the true probability is p.
% This example explores the distribution of the belief weight (b1) for model
% 1 as more tosses are made over time.

%%
p=0.5; p1=0.9; p2=0.1;
T=30; 
reps=100000;
b=zeros(reps,T+1); b(:,1)=0.5; 
for t=1:T 
  x=rand(reps,1)<p;
  l1=p1.^x.*(1-p1).^(1-x);
  l2=p2.^x.*(1-p2).^(1-x);
  bt=l1.*b(:,t);
  bt=bt./(bt+l2.*(1-b(:,t)));
  b(:,t+1)=bt;
end

%%
figure(1); clf
bins=0:0.05:1;
h=hist(b(:,end),0:0.05:1);
bar(bins, h/size(b,1)) % Add '1' for full width
mean(b(:,end)>0.5)
xlabel('b_1(T)')
ylabel('prob(b_1(T))')
xlim([-0.05,1.05])
title(['Histogram for b_1(T), T=' num2str(T)])

%%
%figure(2)
%plot(0:99,b(1:5,1:100))
%xlabel('t')
%ylabel('b_1')
%title(['5 sample paths for belief weight b_1'])