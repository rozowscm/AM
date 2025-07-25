function threemodelfigures(T,eta,fignum)

% Demonstrates the updating of weights on alternative models
% An n-category random variable Y has probability P*w
% where P is a known n x q probability matrix w/ non-negative values 
% and columns that sum to 1 and w is an unknown probability vector with a
% Dirichlet(a) prior. When event i occurs the vector p=P(i,:) provides the
% likelihood used to update the Dirichlet parameters to obtain an
% approximate posterior.

% this file generates a 3-model example

n=6;
q=3;
rep=100000;
P=[exp(n:-1:1)',(n-(n/2))^2-((0:n-1)'-(n-1)/2).^2,exp(1:n)'];
P=normalizeP(P);

%wvalues={[.3;.1;.60],[0.1;0.1;0.80],[0.9;0.05;0.05]};
wvalues={[.3;.1;.60],[0;0;1],[0.5344;0.3594;0.1063]};
pvalues={P*wvalues{1},P*wvalues{2},P*wvalues{3}};
p = pvalues{3} + rand(size(P,1),1)/6;
p = normalizeP(p);
p = [ 0.4104    0.2235    0.0756    0.1369    0.0659    0.0878]';
%not certain what is happening in lines 23 and 24 but I think we are
%choosing a model weight set that is not in the convex combination.
pvalues{3} = p;
% find minKL weighted average model
KLr=@(w) p'*log(p./(P*[w;1-sum(w)]));
w=fminsearch(KLr,wvalues{3}(1:2,end)); w=[w;1-sum(w)];
wvalues{3} = w;
 
sims=length(pvalues);
GAM = zeros(q,sims);
KLB = zeros(1,sims);
KLW = zeros(1,sims);
best=zeros(1,q);

for model=1:sims
  p=pvalues{model};
  GAM(:,model)=(p'*log(P))';      % gamma values for each model
  [~,best(model)]=max(GAM(:,model));
  KLB(model) = p'*log(p./(P(:,best(model)))); 
  KLW(model) = p'*log(p./(P*wvalues{model})); 
end  

GAM, best, KLB, KLW


%% simulation
% set random number generator state to produce the same results each time
%s = RandStream('mt19937ar','Seed',1);

ETAall=cell(1,sims);
KWall=cell(1,sims);
KBall=cell(1,sims);
Wall=cell(1,sims);
Ball=cell(1,sims);
for sim=1:sims
  p = pvalues{sim};
  %RandStream.setGlobalStream(s);

  KL=@(w)p'*log(p./(P*w)); 
  W=zeros(q,T+1);
  B=zeros(q,T+1);
  C=zeros(1,T+1);
  KW=zeros(1,T+1);
  KB=zeros(1,T+1);  
  ETA=zeros(1,T+1);
  a=ones(q,1)*(eta/3);
  b=ones(q,1)/q;
  ETA(1)=sum(a);
  W(:,1)=a/sum(a);
  B(:,1)=b;
  KW(:,1)=KL(W(:,1));

  KB(:,1)=KL(b);

  a=repmat(a,1,rep);
  b=repmat(b,1,rep);
  C(1)=p'*log(sum(P/q,2));
  tic
  for i=1:T
     y=randdisc(p,rand(rep,1));
     ap = modelweightsupdatesim(P(y,:)',a,0);
     infind = any(isinf(ap),1);
     if any(infind)
       ap(:,infind)=NaN;
     end
     a=ap;
     ETA(i+1) = mean(sum(a,1),'omitnan');
     W(:,i+1)=mean(bsxfun(@rdivide,a,sum(a,1)),2,'omitnan');
     KW(i+1)=KL(W(:,i+1));
     b=normalizeP(P(y,:)'.*b);
     c=p'*log(P*b);
     C(i+1)=mean(c);
     B(:,i+1)=mean(b,2);
     KB(i+1)=KL(B(:,i+1));
  end
  toc
  Ball{sim}=B;
  Wall{sim}=W;
  KBall{sim}=KB;
  KWall{sim}=KW;
  ETAall{sim}=ETA;
  fprintf('number of bad paths: %1.0f\n',sum(any(isnan(a),1)))
end
   


%% w plot

MS=12;

G=simplexgrid(3,500,1); 
g=G*P'*log(P); 
KK=bsxfun(@minus,G*(P'*log(p)),g);
[~,ii]=max(g,[],2);

figure(fignum); clf; set(gcf,'defaulttextinterpreter','tex')
set(gcf,'units','normalized','position',[0.354 0.193 0.492 0.712])
colormap([0 1 0;0 0 1;1 0 0])
colormap([0.45;0.6;0.75]*[1 1 1])
%subplot(1,2,1)
beliefplot(G,ii,1:3,'w',.06,.04);
hold on
% plot starting point
[xx,yy]=b2xy([1 1 1]/3);
plot(xx,yy,'w^','markersize',8,'MarkerFaceColor',[1 1 1])
% plot best weights
for sim=1:sims
%  [xx,yy]=b2xy(wvalues{sim}');
 % plot(xx,yy,'wo','markersize',8,'MarkerFaceColor',[1 1 1])
  [bx,by]=b2xy(Ball{sim}');
  h=plot(bx,by,'.k-','linewidth',1.5,'markersize',MS);
  [wx,wy]=b2xy(Wall{sim}');
  plot(wx,wy,'.w-','linewidth',1.5,'markersize',MS)
  [xx,yy]=b2xy(wvalues{sim}');
  plot(xx,yy,'wo','markersize',8,'MarkerFaceColor',[1 1 1])
end
% legend('model 1','model 2','model 3','location','eastoutside')
set(gcf,'name','Minimal KL value as a function of w');
%set(gca,'Children',circshift(get(gca,'Children'),1))
legend('model 1','model 2','model 3','location','eastoutside')

hold off
for sim=1:sims
  [xx,yy]=b2xy(wvalues{sim}');
  if sim==1
    xx=0.375; yy=0.2;
  elseif sim==2
    xx=0.475; yy=0.375;
  else
    xx=0.3; yy=0.02;
  end
  text(xx,yy,['sim ' num2str(sim)])
end

%% figure n+1
figure(fignum+1); clf
set(gcf,'name','Time paths of KL values');
set(gcf,'units','normalized','position',[0.200 0.1 0.650 0.75])
set(gcf,'visible','off')
hold on

for sim=1:sims
  plot(0:T,KBall{sim},'-k','linewidth',5-sim)
end

for sim=1:sims
  plot(0:T,KWall{sim},':k','linewidth',5-sim)
end
hold off

xlim([0 30])
ylim([0 max([KBall{1}(1) KBall{2}(1) KBall{3}(1)])+0.01])
%ylim([0 0.65])
lnames={'sim 1: BMA','sim 2:  BMA','sim 3:  BMA','sim 1: LOP','sim 2: LOP','sim 3: LOP'};
legend(lnames,'location','eastoutside','interpreter','latex')
set(gcf,'visible','on')
xlabel('time','Color',[0 0 0])
ylabel('KL divergence')

if 0  % creates insert
axes('position',[0.175 0.65 0.19 0.3]); 
hold on
for sim=1:sims
  plot(0:T,KBall{sim},'-k','linewidth',5-sim)
end
for sim=1:sims
  plot(0:T,KWall{sim},':k','linewidth',5-sim)
end
hold off
ylim([0.015 .06])
xlim([0 6])
end
%%
figure(fignum+10)
for sim=1:sims
  plot(0:T,ETAall{sim},':k','linewidth',5-sim)
  hold on
end
hold off
xlabel('time')
ylabel('$$\eta$$')
ylimvals=get(gca,'YLim');
ylimvals(1) = 0;
set(gca,'YLim',ylimvals);

%% 
figure(fignum+11); clf; 
set(gcf,'defaulttextinterpreter','tex')
set(gcf,'units','normalized','position',[0.354 0.193 0.492 0.712])

G=simplexgrid(3,100,1); 

tnames={'initial','sim 1','sim 2','sim 3'};

for i=1:4
  if i==1
    a = ones(q,1)*(eta/3);
  else
    a=Wall{i-1}(:,end)*ETAall{i-1}(end);
  end
  subplot(2,2,i)
  pp=pdfdirichlet(G,a);
  indinf=isinf(pp);
  pp(indinf)=NaN;
  [h,xg,yg]=beliefplot(G,pp,[0 42.5],'w',.09,.05);
  hold on
  [xx,yy]=b2xy(a'/sum(a));
  plot(xx,yy,'wo','markersize',8,'MarkerFaceColor',[1 1 1])
  hold off
  text(0.05, 0.85,tnames{i},'Units','Normalized')
  
  pos = get(gca,'outerposition');
  pos(3) = pos(3)*1.275;
  pos(4) = pos(4)*1.275;
  if i>2
    pos(2)=0;
  else
    pos(2)=0.5;
  end
  if i==1 || i==3
    pos(1)=0;
  else
    pos(1)=0.5;
  end
  set(gca,'outerposition',pos);
  h=colorbar;
  if 1
  pos = get(h,'position');
  pos(1)=pos(1)+0.8*pos(3);
  pos(3)=pos(3)/2;
  pos(2)=pos(2)+0.06;
  pos(4)=pos(4)*0.9;
  set(h,'position',pos);
  end
  % display infinite PDF values in red
  patch(xg(:,indinf),yg(:,indinf),'r','Edgecolor','none','linestyle','none');
end