% Gouldian finch AM with 3 experts
clear
clc
%%
% parameter values
delta = 0.97; % discount rate
E = 20; % value of being in the high state
C = [0;5;5;5];  % treatment cost

% --- Define potential true model weights ---
w=cell(1,4);
w{1}=[0.33; 0.33; 0.34]; % Approx center
w{2}=[0.20; 0.19; 0.61]; % High weight on expert 3
w{3}=[0.01; 0.20; 0.79]; % Very high weight on expert 3
w{4}=[0.49; 0.49; 0.02]; % Low weight on expert 3

% --- OPTIMIZATION: Define ONLY the eta values needed for plotting ---
etavals = [0.01; 1.5; 3.0; 50]; % Column vector of required etas
inc = 100;  % number of belief intervals for each expert

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

%% Solve BMA
disp('Setting up the AM problem using BMA')
[b,Pb,Rb,Sb,Xb,Ixb] = amdp(inc,P,R,S,X,Ix);
model_bma = struct('P',Pb,'R',Rb,'X',Xb,'Ix',Ixb,'d',delta);
moptions_solve = struct('print',0,'algorithm','i');
disp('Solving BMA MDP...')
results_bma = mdpsolve(model_bma, moptions_solve);
Xopt_bma = results_bma.Xopt;
disp('BMA Solved.')

%% Solve KL Method for Specific Etas
MM_flag=0; % KL Method
disp('Setting up the Dirichlet AM problem using KL for specific etas')
[w_eta_kl,Pw_kl,Ra_kl,Sa_kl,Xa_kl,Ixa_kl] = amdpdirichlet(inc,P,R,S,X,Ix,etavals,MM_flag);
modeldirKL = struct('P',Pw_kl,'R',Ra_kl,'X',Xa_kl,'Ix',Ixa_kl,'d',delta);
disp('Solving Dirichlet KL MDP (reduced eta set)...')
resultsdirKL = mdpsolve(modeldirKL, moptions_solve);
Xopt_dir_kl = resultsdirKL.Xopt;
disp('Dirichlet KL Solved.')

%% PLOTTING (Unique Markers for Points)

% --- Common plotting parameters ---
cmap = [0;0.5;0.7;0.9]*[1 1 1];
ll1 = 0.09; ll2 = 0.07;
action_labels = {'Do Nothing','Fire & Grazing','Control Cats','Nesting Boxes'};
plot_fontsize = 14;
common_ylabel_fontsize = 16;
legend_fontsize = 12;
title_y_offset = 1.08;
label_low_pos  = [0.12, 0.60, 0.15, 0.05];
label_high_pos = [0.12, 0.15, 0.15, 0.05];
manual_legend_pos = [0.33, 0.85, 0.10, 0.07];
point_marker_size = 7; % Slightly larger for visibility

% --- Marker styles for the 4 points --- <<< NEW DEFINITIONS >>>
point_markers = {'o', 's', 'd', '^'}; % Circle, Square, Diamond, Triangle
point_face_colors = {'k', [0.8500 0.3250 0.0980], [0.4660 0.6740 0.1880], [0 0.4470 0.7410]}; % Black, Orange/Red, Green, Blue
point_edge_color = 'k'; % Use black edge for all for contrast

% --- Function handle for barycentric to Cartesian conversion ---
baryToCart = @(w_vec) [w_vec(2) + 0.5*w_vec(3), w_vec(3)*sqrt(3)/4]; % Using adjusted Y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE 1: Compare BMA, KL eta=0.01, KL eta=50
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Generating Figure 1...')
figure(1); clf;
current_fig1 = gcf;
set(current_fig1,'units','normalized','position',[0.01 0.1 0.98 0.8],'color','w')
set(current_fig1, 'DefaultTextInterpreter', 'none');
colormap(cmap)

target_eta1 = 0.01;
target_eta2 = 50;

% --- Row 1: Low State (State == 0) ---
subplot(2,3,1);
ind = Xopt_bma(:,2)==0; Xopt_b=Xopt_bma(ind,3:5); Xopt_a=Xopt_bma(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
t = title('BMA','FontSize', plot_fontsize, 'Interpreter', 'none'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,2);
ind_kl = (Xopt_dir_kl(:,2)==0) & (abs(Xopt_dir_kl(:,6) - target_eta1) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta1)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,3);
ind_kl = (Xopt_dir_kl(:,2)==0) & (abs(Xopt_dir_kl(:,6) - target_eta2) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta2)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);


% --- Row 2: High State (State == 1) ---
subplot(2,3,4);
ind = Xopt_bma(:,2)==1; Xopt_b=Xopt_bma(ind,3:5); Xopt_a=Xopt_bma(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
t = title('BMA','FontSize', plot_fontsize, 'Interpreter', 'none'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,5);
ind_kl = (Xopt_dir_kl(:,2)==1) & (abs(Xopt_dir_kl(:,6) - target_eta1) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta1)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,6);
ind_kl = (Xopt_dir_kl(:,2)==1) & (abs(Xopt_dir_kl(:,6) - target_eta2) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta2)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

% --- Legend: Manually Positioned ---
hL1 = legend(action_labels, 'Orientation', 'vertical', 'FontSize', 16, 'Interpreter', 'none', 'Location', 'none');
set(hL1, 'Units', 'normalized'); set(hL1, 'Position', manual_legend_pos);

% --- Common YLabel (Using Annotation) ---
annotation(current_fig1,'textbox', label_low_pos, 'String', 'Low State', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 90, 'FontWeight', 'bold', 'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');
annotation(current_fig1,'textbox', label_high_pos, 'String', 'High State', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 90, 'FontWeight', 'bold', 'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE 2: Compare BMA, KL eta=1.5, KL eta=3.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Generating Figure 2...')
figure(2); clf;
current_fig2 = gcf;
set(current_fig2,'units','normalized','position',[0.02 0.15 0.96 0.8],'color','w')
set(current_fig2, 'DefaultTextInterpreter', 'none');
colormap(cmap)

target_eta3 = 1.5;
target_eta4 = 3.0;

% --- Row 1: Low State (State == 0) ---
subplot(2,3,1);
ind = Xopt_bma(:,2)==0; Xopt_b=Xopt_bma(ind,3:5); Xopt_a=Xopt_bma(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
t = title('BMA','FontSize', plot_fontsize, 'Interpreter', 'none'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,2);
ind_kl = (Xopt_dir_kl(:,2)==0) & (abs(Xopt_dir_kl(:,6) - target_eta3) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta3)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,3);
ind_kl = (Xopt_dir_kl(:,2)==0) & (abs(Xopt_dir_kl(:,6) - target_eta4) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta4)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

% --- Row 2: High State (State == 1) ---
subplot(2,3,4);
ind = Xopt_bma(:,2)==1; Xopt_b=Xopt_bma(ind,3:5); Xopt_a=Xopt_bma(ind,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'b',ll1,ll2); end
t = title('BMA','FontSize', plot_fontsize, 'Interpreter', 'none'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,5);
ind_kl = (Xopt_dir_kl(:,2)==1) & (abs(Xopt_dir_kl(:,6) - target_eta3) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta3)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

subplot(2,3,6);
ind_kl = (Xopt_dir_kl(:,2)==1) & (abs(Xopt_dir_kl(:,6) - target_eta4) < 1e-6); Xopt_b=Xopt_dir_kl(ind_kl,3:5); Xopt_a=Xopt_dir_kl(ind_kl,1);
if ~isempty(Xopt_a); beliefplot(Xopt_b,Xopt_a,A,'w',ll1,ll2); end
t = title(['\eta = ', num2str(target_eta4)],'FontSize', plot_fontsize, 'Interpreter', 'tex'); set(t, 'Units', 'normalized', 'Position', [0.5, title_y_offset, 0]);
hold on; for i_w = 1:length(w); coords = baryToCart(w{i_w}); plot(coords(1), coords(2), 'Marker', point_markers{i_w}, 'MarkerEdgeColor', point_edge_color, 'MarkerFaceColor', point_face_colors{i_w}, 'MarkerSize', point_marker_size, 'LineStyle', 'none'); end; hold off; % Use indexed markers/colors
xlim([-0.05, 1.05]); ylim([-0.05, 0.5]);

% --- Legend: Manually Positioned ---
hL2 = legend(action_labels, 'Orientation', 'vertical', 'FontSize', 16, 'Interpreter', 'none', 'Location', 'none');
set(hL2, 'Units', 'normalized'); set(hL2, 'Position', manual_legend_pos); % Use same manual position

% --- Common YLabel (Using Annotation) ---
annotation(current_fig2,'textbox', label_low_pos, 'String', 'Low State', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 90, 'FontWeight', 'bold', 'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');
annotation(current_fig2,'textbox', label_high_pos, 'String', 'High State', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 90, 'FontWeight', 'bold', 'FontSize', common_ylabel_fontsize, 'Interpreter', 'none');

disp('Processing Complete. Figures 1 and 2 generated with unique markers for points.');

% NOTE: Assumes the 'beliefplot' function is accessible in your MATLAB path.
%       Also assumes 'amdp', 'mdpsolve', 'amdpdirichlet', 'rectgrid', 'getI' functions are available.