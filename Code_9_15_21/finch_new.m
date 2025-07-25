% Gouldian finch AM with 3 experts
clear
clc
%%
% parameter values
delta = 0.99; % discount rate
E = 20; % value of being in the high state
C = [0;5;5;5];  % treatment cost

% --- MINIMAL CHANGE 1: Optimize etavals ---
etavals = [0.01; 1.5; 3.0; 50]; % Column vector of required etas
inc = 100;  % number of belief intervals for each expert
% maxEta= 50; % Keep if needed, maybe redundant now

% state and action values
S = [0;1]; % low,high
A = [1;2;3;4]; % actions 1-Do Nothing, 2-Fire and grazing management, 3-Control feral cats, 4-Provide nesting boxes
X = rectgrid(A,S); % X order: action then state
svars = 2; % column of X with the state variable
Ix = getI(X,svars); % index vector that simply maps state/actions to states
R = E*X(:,2)-C(X(:,1)); % reward for high state less cost of treatment

% probability transition matrix. 3 experts.
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

%% Solve BMA (Original Code)
disp('Setting up the AM problem using BMA')
[b,Pb,Rb,Sb,Xb,Ixb] = amdp(inc,P,R,S,X,Ix);
model_bma = struct('P',Pb,'R',Rb,'X',Xb,'Ix',Ixb,'d',delta); % Use model_bma to avoid conflict
moptions_solve=struct('print',0,'algorithm','i'); %% Supress print output
disp('Solving BMA MDP...')
results_bma = mdpsolve(model_bma,moptions_solve); % Use results_bma
Xopt = results_bma.Xopt; % Keep original Xopt name for plotting compatibility
disp('BMA Solved.')


%% Solve KL Method (Original Code, but using optimized etavals)
MM=0; %KL Method flag from original code
disp('Setting up the Dirichlet AM problem using KL')
[w_eta2,Pw,Ra,Sa2,Xa,Ixa] = amdpdirichlet(inc,P,R,S,X,Ix,etavals,MM); % Use optimized etavals
modeldirKL = struct('P',Pw,'R',Ra,'X',Xa,'Ix',Ixa,'d',delta);
disp('Solving Dirichlet KL MDP...')
resultsdirKL = mdpsolve(modeldirKL,moptions_solve); % Use moptions_solve
XoptdirKL = resultsdirKL.Xopt; % Keep original XoptdirKL name
disp('Dirichlet KL Solved.')


%% Plotting Section - Modified Figure 1, New Figure 2

action_labels = {'Do Nothing','Fire & Grazing','Control Cats','Nesting Boxes'}; % Define once
manual_legend_pos = [0.33, 0.85, 0.10, 0.07]; % STARTING GUESS - ADJUST AS NEEDED
common_ylabel_fontsize = 16; % Font size for the shared Y label

% Annotation positions [X, Y, Width, Height] in normalized FIGURE units - Adjust Y values!
label_low_pos  = [0.12, 0.60, 0.15, 0.05]; % [X, Y, W, H] for "Low State" (Higher Y)
label_high_pos = [0.12, 0.15, 0.15, 0.05]; % [X, Y, W, H] for "High State" (Lower Y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE 1: Compare BMA, KL eta=0.01, KL eta=50 (Adapted from Original Fig 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Generating Figure 1...')
ll1=0.09; ll2=0.07; % Layout params from original
figure(1); clf;
set(gcf,'units','normalized','position',[0.01 0.1 0.98 0.8]) % Slightly adjusted position
set(gcf,'color',[1 1 1])
colormap([0;0.5;0.7;0.9]*[1 1 1]) % From original

target_eta1 = 0.01;
target_eta2 = 50;

% --- Row 1: Low State (State == 0) ---
subplot(2,3,1); % Changed layout to 2x3
ind = Xopt(:,2)==0; % Use original BMA variable Xopt
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
title({'BMA';''},'FontSize', 14); % Original title style

subplot(2,3,2); % New plot
ind_kl = (XoptdirKL(:,2)==0) & (abs(XoptdirKL(:,6) - target_eta1) < 1e-6); % Filter KL results
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end % Use 'w' style like original MM/KL plots
title({['\eta = ', num2str(target_eta1)];''},'FontSize', 14, 'Interpreter', 'tex'); % Original title style

subplot(2,3,3); % New plot (potentially empty box)
ind_kl = (XoptdirKL(:,2)==0) & (abs(XoptdirKL(:,6) - target_eta2) < 1e-6);
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta2)];''},'FontSize', 14, 'Interpreter', 'tex');

% --- Row 2: High State (State == 1) ---
subplot(2,3,4); % Changed layout from 2,4,5
ind = Xopt(:,2)==1; % Use original BMA variable Xopt
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
title({'BMA';''},'FontSize', 14);

subplot(2,3,5); % New plot
ind_kl = (XoptdirKL(:,2)==1) & (abs(XoptdirKL(:,6) - target_eta1) < 1e-6);
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta1)];''},'FontSize', 14, 'Interpreter', 'tex');

subplot(2,3,6); % New plot
ind_kl = (XoptdirKL(:,2)==1) & (abs(XoptdirKL(:,6) - target_eta2) < 1e-6);
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta2)];''},'FontSize', 14, 'Interpreter', 'tex');

% --- Legend: Manually Positioned ---
hL1 = legend(action_labels, ...
            'Orientation', 'vertical', ... % Ensure vertical
            'FontSize', 16, ...
            'Interpreter', 'none', ...
            'Location', 'none');      % Create with no specific location first
set(hL1, 'Units', 'normalized');     % Set units to figure-normalized
set(hL1, 'Position', manual_legend_pos); % Apply manual position relative to figure

% --- Common YLabel (Using Annotation) ---
annotation(figure(1),'textbox', label_low_pos, ...
           'String', 'Low State', 'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'Rotation', 90, 'FontWeight', 'bold', ...
           'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');
annotation(figure(1),'textbox', label_high_pos, ...
           'String', 'High State', 'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'Rotation', 90, 'FontWeight', 'bold', ...
           'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE 2: Compare BMA, KL eta=1.5, KL eta=3.0 (Mimicking Fig 1 Style)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Generating Figure 2...')
ll1=0.09; ll2=0.07; % Layout params
figure(2); clf; % New figure number
set(gcf,'units','normalized','position',[0.02 0.15 0.96 0.8]) % Adjusted position slightly
set(gcf,'color',[1 1 1])
colormap([0;0.5;0.7;0.9]*[1 1 1])

target_eta3 = 1.5;
target_eta4 = 3.0;

% --- Row 1: Low State (State == 0) ---
subplot(2,3,1);
ind = Xopt(:,2)==0; % Use original BMA variable Xopt
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
title({'BMA';''},'FontSize', 14);

subplot(2,3,2);
ind_kl = (XoptdirKL(:,2)==0) & (abs(XoptdirKL(:,6) - target_eta3) < 1e-6); % Filter KL results
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta3)];''},'FontSize', 14, 'Interpreter', 'tex');

subplot(2,3,3);
ind_kl = (XoptdirKL(:,2)==0) & (abs(XoptdirKL(:,6) - target_eta4) < 1e-6);
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta4)];''},'FontSize', 14, 'Interpreter', 'tex');

% --- Row 2: High State (State == 1) ---
subplot(2,3,4);
ind = Xopt(:,2)==1; % Use original BMA variable Xopt
Xopt_b=Xopt(ind,3:5);
Xopt_a=Xopt(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
title({'BMA';''},'FontSize', 14);

subplot(2,3,5);
ind_kl = (XoptdirKL(:,2)==1) & (abs(XoptdirKL(:,6) - target_eta3) < 1e-6);
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta3)];''},'FontSize', 14, 'Interpreter', 'tex');

subplot(2,3,6);
ind_kl = (XoptdirKL(:,2)==1) & (abs(XoptdirKL(:,6) - target_eta4) < 1e-6);
Xopt_b=XoptdirKL(ind_kl,3:5);
Xopt_a=XoptdirKL(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
title({['\eta = ', num2str(target_eta4)];''},'FontSize', 14, 'Interpreter', 'tex');

% --- Legend: Manually Positioned ---
hL2 = legend(action_labels, ...
            'Orientation', 'vertical', ... % Ensure vertical
            'FontSize', 16, ...
            'Interpreter', 'none', ...
            'Location', 'none');      % Create with no specific location first
set(hL2, 'Units', 'normalized');     % Set units to figure-normalized
set(hL2, 'Position', manual_legend_pos); % Apply manual position relative to figure (use same variable for consistency)

% --- Common YLabel (Using Annotation) ---
annotation(figure(2),'textbox', label_low_pos, ...
           'String', 'Low State', 'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'Rotation', 90, 'FontWeight', 'bold', ...
           'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');
annotation(figure(2),'textbox', label_high_pos, ...
           'String', 'High State', 'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
           'Rotation', 90, 'FontWeight', 'bold', ...
           'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');

disp('Processing Complete.');