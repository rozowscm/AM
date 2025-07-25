% Gouldian finch AM with 3 experts
clear
clc

%%
% parameter values
delta = 0.98; % discount rate
E = 20; % value of being in the high state
C = [0;5;5;5];  % treatment cost
%minimum value for sum(alpha_i)= 1/#models is a guide.
maxEta= 50; % maximum value for sum(alpha_i) 
inc = 100;  % number of belief intervals for each expert
etavals=[0.01 0.025 0.05 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 linspace(2,maxEta,193)]';

% state and action values
S = [0;1]; % low,high
A = [1;2;3;4]; % actions 1-Do Nothing, 2-Fire and grazing management, 3-Control feral cats, 4-Provide nesting boxes
X = rectgrid(A,S); % X order: action then state
svars = 2; % column of X with the state variable
Ix = getI(X,svars); % index vector that simply maps state/actions to states
R = E*X(:,2)-C(X(:,1)); % reward for high state less cost of treatment

% probability transition matrix. 3 experts.
% P: each cell element is for a different expert
% Each cell is a set of 4 2x2 matrices stacked horizontally
P = cell(1,3);
P{1} = [...
0.99 0.50 0.20 0.05 0.40 0.40 0.60 0.40 
0.01 0.50 0.80 0.95 0.60 0.60 0.40 0.60 ];
P{2} = [...
0.90 0.60 0.55 0.30 0.20 0.10 0.40 0.25 
0.10 0.40 0.45 0.70 0.80 0.90 0.60 0.75 ];
P{3} = [...
0.80 0.50 0.70 0.40 0.70 0.30 0.50 0.20 
0.20 0.50 0.30 0.60 0.30 0.70 0.50 0.80 ];


%%
disp('Setting up the AM problem using BMA')
[b,Pb,Rb,Sb,Xb,Ixb] = amdp(inc,P,R,S,X,Ix);
model = struct('P',Pb,'R',Rb,'X',Xb,'Ix',Ixb,'d',delta);
moptions=struct('print',2,'algorithm','i'); %%display iteration report, algorithm=policy iteration-infinite
results = mdpsolve(model,moptions);
Xopt = results.Xopt;
%{
MM=1; %CDS Method
disp('Setting up the Dirichlet AM problem using MM')
[w_eta,Pw,Ra,Sa,Xa,Ixa] = amdpdirichlet(inc,P,R,S,X,Ix,etavals,MM);
modeldirMM = struct('P',Pw,'R',Ra,'X',Xa,'Ix',Ixa,'d',delta);
resultsdirMM = mdpsolve(modeldirMM,moptions);
XoptdirMM = resultsdirMM.Xopt;
%}

MM=0; %KL Method
disp('Setting up the Dirichlet AM problem using KL')
[w_eta2,Pw,Ra,Sa2,Xa,Ixa] = amdpdirichlet(inc,P,R,S,X,Ix,etavals,MM);
modeldirKL = struct('P',Pw,'R',Ra,'X',Xa,'Ix',Ixa,'d',delta);
resultsdirKL = mdpsolve(modeldirKL,moptions);
XoptdirKL = resultsdirKL.Xopt;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compare BMA to MM-LOP at various etas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set(0,'DefaultTextInterpreter','none','DefaultTextFontSize',16,'DefaultAxesFontSize',16)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under BMA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = Xopt(:,2)==0; 
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
ll1=0.09; ll2=0.07;
figure(1); clf; 
set(gcf,'units','normalized','position',[0.003 0.069 0.989 0.694])
set(gcf,'color',[1 1 1])
subplot(2,4,1);
colormap([0;0.5;0.7;0.9]*[1 1 1])
beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2);
title({'BMA';''},'FontSize', 14);

ind = Xopt(:,2)==1; 
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
%figure(2); clf;
subplot(2,4,5);
beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2);
title({'BMA';''},'FontSize', 14);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under updated weights MM compare etas.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
etas=unique(w_eta(:,numel(P)+1));

%%%%Choose min eta%%%%
mineta=min(etas);
ind = XoptdirMM(:,6)==mineta;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,2);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['\eta = ', num2str(mineta)];''});

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,6);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['\eta = ', num2str(mineta)];''},'FontSize', 14);

%%%%Choose max eta%%%%
maxeta=max(etas);
ind = XoptdirMM(:,6)==maxeta;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1); 
subplot(2,4,3);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['\eta = ', num2str(maxeta)];''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,7);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['\eta = ', num2str(maxeta)];''},'FontSize', 14);

%%%%Solve for eta=inf%%%%
PP=b(:,1)*P{1}(2,:); for i=2:length(P), PP=PP+b(:,i)*P{i}(2,:); end; 
nb = size(b,1);
PP1=reshape(PP,[nb 1 8]);
PP=zeros(nb, 2, 8);
PP(:,1,:)=1-PP1;
PP(:,2,:)=PP1;
clear PP1

AA=zeros(nb,2);
model=struct('P',[],'R',R,'X',X,'Ix',Ix,'d',delta);
moptions.print=0;
for i=1:nb
  model.P=squeeze(PP(i,:,:));
  results = mdpsolve(model,moptions);
  AA(i,:) = results.Xopt(:,1)';
end

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
subplot(2,4,4);
beliefplot(Xoptdir_b,AA(:,1),A,'w',ll1,ll2);
title({'\eta = \infty';''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
subplot(2,4,8);
beliefplot(Xoptdir_b,AA(:,2),A,'w',ll1,ll2);
title({['\eta = \infty'];''},'FontSize', 14);

h=legend('Do Nothing','Fire & Grazing','Control Cats','Nesting Boxes','location','southoutside','FontSize', 14);
pos=get(h,'position');
pos(1)=1-5.8*pos(3);
pos(2)=2/3+1.1*pos(4);
set(h,'position',pos)

% Give common xlabel, ylabel and title to your figure
han=axes(figure(1),'visible','off'); 
han.YLabel.Visible='on';
ylabel(han,'High State                                     Low State','fontweight','bold','FontSize', 16);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compare BMA to MM-LOP at suggested etas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under BMA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = Xopt(:,2)==0; 
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
ll1=0.09; ll2=0.07;
figure(2); clf; 
set(gcf,'units','normalized','position',[0.003 0.069 0.989 0.694])
set(gcf,'color',[1 1 1])
subplot(2,4,1);
colormap([0;0.5;0.7;0.9]*[1 1 1])
beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2);
title({'BMA';''},'FontSize', 16);

ind = Xopt(:,2)==1; 
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
subplot(2,4,5);
beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2);
title({'BMA';''},'FontSize', 16);

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under updated weights MM compare suggested initial etas.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Choose Jeffreys%%%%
ind = XoptdirMM(:,6)==1.5;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,2);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({'\eta = 1.5';''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,6);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({'\eta = 1.5';''},'FontSize', 14);

%%%%Choose Uniform%%%%
ind = XoptdirMM(:,6)==3;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1); 
subplot(2,4,3);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({'\eta = 3.0';''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,7);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({'\eta = 3.0';''},'FontSize', 14);

%%%%Choose 2%%%%
ind = XoptdirMM(:,6)==6;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,4);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({'\eta = 6.0';''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,4,8);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({'\eta = 6.0';''},'FontSize', 14);


h=legend('Do Nothing','Fire & Grazing','Control Cats','Nesting Boxes','location','southoutside','FontSize', 14);
pos=get(h,'position');
pos(1)=1-5.8*pos(3);
pos(2)=2/3+1.1*pos(4);
set(h,'position',pos)

% Give common xlabel, ylabel and title to your figure
han=axes(figure(2),'visible','off'); 
han.YLabel.Visible='on';
ylabel(han,'High State                                           Low State','fontweight','bold','FontSize', 16);



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compare BMA to MM and KL optimal actions under small etas.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = Xopt(:,2)==0; 
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
ll1=0.09; ll2=0.07;
figure(3); clf; 
set(gcf,'units','normalized','position',[0.003 0.069 0.989 0.694])
set(gcf,'color',[1 1 1])
subplot(2,3,1);
colormap([0;0.5;0.7;0.9]*[1 1 1])
beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2);
title({'BMA';''},'FontSize', 14);

ind = Xopt(:,2)==1; 
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
%figure(2); clf;
subplot(2,3,4);
beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2);
title({'BMA';''},'FontSize', 14);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under updated weights MM compare etas.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
etas=unique(w_eta(:,numel(P)+1));
%%%%Choose min eta%%%%
mineta=min(etas);
ind = XoptdirMM(:,6)==mineta;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,3,2);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['MM with \eta = ', num2str(mineta)];''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,3,5);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['MM with \eta = ', num2str(mineta)];''},'FontSize', 14);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under updated weights KL compare etas.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
etas=unique(w_eta(:,numel(P)+1));
%%%%Choose min eta%%%%
mineta=min(etas);
ind = XoptdirKL(:,6)==mineta;
Xoptdir_eta=XoptdirKL(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,3,3);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['KL with \eta = ', num2str(mineta)];''},'Interpreter', 'tex','FontSize', 16);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,3,6);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['KL with \eta = ', num2str(mineta)];''},'Interpreter', 'tex','FontSize', 16);

h=legend('Do Nothing','Fire & Grazing','Control Cats','Nesting Boxes','location','southoutside','FontSize', 16);
pos=get(h,'position');
pos(1)=1-5.8*pos(3);
pos(2)=2/3+1.1*pos(4);
set(h,'position',pos)

% Give common xlabel, ylabel and title to your figure
han=axes(figure(3),'visible','off'); 
han.YLabel.Visible='on';
ylabel(han,'High State                                     Low State','fontweight','bold','FontSize', 18);


%{
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compare MM and KL optimal actions under Jeffrey's Prior.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ll1=0.09; ll2=0.07;
figure(4); clf; 
set(gcf,'units','normalized','position',[0.003 0.069 0.989 0.694])
set(gcf,'color',[1 1 1])
colormap([0;0.5;0.7;0.9]*[1 1 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under updated weights MM compare Jeffreys.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = XoptdirMM(:,6)==1.5;
Xoptdir_eta=XoptdirMM(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,2,1);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['MM with \eta = ', num2str(1.5)];''},'FontSize', 14);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,2,3);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['MM with \eta = ', num2str(1.5)];''},'FontSize', 14);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot optimal actions under updated weights KL compare Jeffreys.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = XoptdirKL(:,6)==1.5;
Xoptdir_eta=XoptdirKL(ind,:);

ind = Xoptdir_eta(:,2)==0; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,2,2);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['KL with \eta = ', num2str(1.5)];''},'Interpreter', 'tex','FontSize', 16);

ind = Xoptdir_eta(:,2)==1; 
Xoptdir_b = Xoptdir_eta(ind,3:5);
Xoptdir_a = Xoptdir_eta(ind,1);
subplot(2,2,4);
beliefplot(Xoptdir_b,Xoptdir_a,A,'w',ll1,ll2);
title({['KL with \eta = ', num2str(1.5)];''},'Interpreter', 'tex','FontSize', 16);

h=legend('Do Nothing','Fire & Grazing','Control Cats','Nesting Boxes','location','southoutside','FontSize', 16);
pos=get(h,'position');
pos(1)=1-5.0*pos(3);
pos(2)=2/3+.9*pos(4);
set(h,'position',pos)

% Give common xlabel, ylabel and title to your figure
han=axes(figure(4),'visible','off'); 
han.YLabel.Visible='on';
ylabel(han,'High State                                     Low State','fontweight','bold','FontSize', 18);
