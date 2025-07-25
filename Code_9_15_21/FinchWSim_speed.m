%% Set up
reps=100000;
T = 100; % Max simulation time
s=2;
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

q = size(P{1}, 2);
num_models = length(P);
if num_models ~= 3
    warning('Code assumes 3 models (q=3). Review storage logic if q~=3.');
end
q = num_models; % Should be 3

% Define potential true model weights
w=cell(1,4);
w{1}=[0.2046; 0.3732; 0.4223];
w{2}=[0.0165; 0.6101; 0.3734];
w{3}=[0.6220; 0.2938; 0.0842];
w{4}=[0.0000; 0.0000; 1.0000];


% Define potential true model weights
w=cell(1,4);
w{1}=[0.33; 0.33; 0.34];
w{2}=[0.20; 0.19; 0.61];
w{3}=[0.01; 0.20; 0.79];
w{4}=[0.49; 0.49; 0.02];
w{4}=[0.0000; 0.0000; 1.0000];

% Define alpha multipliers (drop 0.01)
v = [0.1 0.5 1];
num_alpha_levels = length(v); % Will be 4

% Define specific time points for summary
summary_times = [5, 10, 15, 20];
num_summary_times = length(summary_times);

%% Pre-allocate Results Cell Arrays
vals_summary = cell(4, num_alpha_levels, num_summary_times); % 4x4x5
vals = cell(4, num_alpha_levels); % 4x4

%% Combined Simulation and Processing Loop using PARFOR
fprintf('Running combined simulations and processing in parallel...\n');
delete(gcp('nocreate'));
pool = parpool(8); % Adjust worker count as needed

parfor i = 1:4
    wstar_i = w{i};
    if size(wstar_i, 2) ~= 1; wstar_i = wstar_i'; end
    fprintf(' Starting processing for wstar set %d (Worker assignment may vary)\n', i);

    vals_i = cell(1, num_alpha_levels);
    vals_summary_i = cell(1, num_alpha_levels, num_summary_times);
    Afunc_i = @(S,Alpha) ones(size(S,1),1);

    for j = 1:num_alpha_levels
        Alpha_0 = ones(1, q) * v(j);
        S_0 = repmat(s, reps, 1);
        b_0 = ones(1, q) * 1/q;
        eta_0 = sum(Alpha_0);

        [S, A, Alpha, Sb, Ab, B] = amdpWsim(P, wstar_i, T, S_0, Alpha_0, b_0, Afunc_i, 0);
        actual_T_sim = size(Alpha, 3);

        Alphas_path = squeeze(mean(Alpha, 1));
        bs_path = squeeze(mean(B, 1));
        ws_path = normalizeP(Alphas_path);
        vals_i{j} = [wstar_i bs_path ws_path];

        for k_idx = 1:num_summary_times
            l = summary_times(k_idx);
            time_idx = l; % Assuming 1-based indexing

            if time_idx < 1 || time_idx > actual_T_sim
               continue;
            end

            AlphaTl = mean(Alpha(:,:,time_idx), 1);
            bTl = mean(B(:,:,time_idx), 1);
            wTl = normalizeP(AlphaTl')';
            stdbTl = std(B(:,:,time_idx), 0, 1);
            w1Tl_per_rep = normalizeP(squeeze(Alpha(:,:,time_idx))');
            stdwTl = std(w1Tl_per_rep, 0, 2);

            if size(bTl,1) < size(bTl,2); bTl = bTl'; end
            if size(wTl,1) < size(wTl,2); wTl = wTl'; end
            if size(stdbTl,1) < size(stdbTl,2); stdbTl = stdbTl'; end
            if size(stdwTl,1) < size(stdwTl,2); stdwTl = stdwTl'; end

            etaTl = sum(AlphaTl);
            if q == 3
                etas_summary = [0; eta_0; etaTl];
            else
                etas_summary = [eta_0; etaTl; NaN(q-2, 1)];
            end
            t_summary = repmat(l, q, 1);

            vals_summary_i{1, j, k_idx} = [t_summary wstar_i bTl stdbTl wTl stdwTl etas_summary(1:q,:)];
        end % k_idx loop
    end % j loop
    vals(i,:) = vals_i;
    vals_summary(i,:,:) = vals_summary_i;
end % parfor i loop
fprintf('Finished combined simulations and processing.\n');


%% ======== NEW DISPLAY SECTION FOR LATEX-LIKE TABLE OUTPUT ================

fprintf('\n\n--- Summary Tables ---\n');

% Define fixed column widths for formatting consistency
colW_m = -8; % Model name (left align)
colW_d = 10; % Data columns (right align)
colW_s = 8; % Std Dev columns (in parentheses, right align within space)
prec = 4;  % Precision (decimal places)

for i = 1:4 % Loop through w* sets
    wstar_disp = w{i}; % Get the true weights for this set

    for k_idx = 1:num_summary_times % Loop through time points
        current_time = summary_times(k_idx);
        fprintf('\n================================================================================\n');
        fprintf(' Table for w* set %d, Time = %d\n', i, current_time);
        fprintf('================================================================================\n');

        % --- Aggregate data for this table ---
        % Check if data exists for this time point for j=1 (needed for True/BMA)
        if isempty(vals_summary{i,1,k_idx}) || size(vals_summary{i,1,k_idx},2) ~= 7
             fprintf('Data for w* set %d, Time %d not available or invalid. Skipping table.\n', i, current_time);
             continue; % Skip this time point if essential data is missing
        end

        true_w = vals_summary{i,1,k_idx}(:, 2);
        bma_mean = vals_summary{i,1,k_idx}(:, 3); % Using j=1 for BMA (Mean)
        bma_std = vals_summary{i,1,k_idx}(:, 4);  % Using j=1 for BMA (Std Dev)

        lop_means = zeros(q, num_alpha_levels);
        lop_stds = zeros(q, num_alpha_levels);
        eta0_vals = zeros(1, num_alpha_levels);
        etaT_vals = zeros(1, num_alpha_levels);
        data_valid = true; % Flag to check if all needed data is present

        for j = 1:num_alpha_levels
            if isempty(vals_summary{i,j,k_idx}) || size(vals_summary{i,j,k_idx},2) ~= 7
                fprintf('LOP data for w* set %d, Time %d, Alpha factor %.2f (j=%d) not available or invalid.\n', i, current_time, v(j), j);
                data_valid = false;
                lop_means(:,j) = NaN; % Mark as NaN
                lop_stds(:,j) = NaN;
                eta0_vals(j) = NaN;
                etaT_vals(j) = NaN;
                continue; % Skip this alpha level if data missing
            end
            lop_means(:,j) = vals_summary{i,j,k_idx}(:, 5); % LOP Mean (col 5)
            lop_stds(:,j) = vals_summary{i,j,k_idx}(:, 6);  % LOP Std Dev (col 6)
            eta0_vals(j) = q * v(j); % Calculate initial eta
             % Retrieve stored etaT
             eta_col = vals_summary{i,j,k_idx}(:, 7);
             if q == 3 && size(eta_col,1)>=3
                 etaT_vals(j) = eta_col(3);
             elseif q ~= 3 && size(eta_col,1)>=2
                 etaT_vals(j) = eta_col(2);
             else
                 etaT_vals(j) = NaN; % Cannot retrieve
             end
        end

        % Retrieve eta_0 and eta_T for the 'BMA' column (using j=1 results)
        eta0_bma = q * v(1);
        etaT_bma = NaN;
        if ~isempty(vals_summary{i,1,k_idx}) && size(vals_summary{i,1,k_idx},2)==7
            eta_col_bma = vals_summary{i,1,k_idx}(:, 7);
             if q == 3 && size(eta_col_bma,1)>=3
                 etaT_bma = eta_col_bma(3);
             elseif q ~= 3 && size(eta_col_bma,1)>=2
                 etaT_bma = eta_col_bma(2);
             end
        end


        % --- Print Table Headers ---
        fprintf('%*s %*s %*s ', colW_m, 'Model', colW_d, 'True', colW_d, 'BMA');
        % Print LOP headers dynamically
        for j = 1:num_alpha_levels
            fprintf('%*s ', colW_d, sprintf('LOP(a=%.1f)', v(j))); % Use alpha value in header
        end
        fprintf('\n'); % Newline after header

        % --- Print Model Rows ---
        for model_idx = 1:q
            % Print Mean Row
            fprintf('%*s %*.*f %*.*f ', colW_m, sprintf('Expert %d', model_idx), ...
                colW_d, prec, true_w(model_idx), ...
                colW_d, prec, bma_mean(model_idx));
            for j = 1:num_alpha_levels
                fprintf('%*.*f ', colW_d, prec, lop_means(model_idx, j));
            end
            fprintf('\n');

            % Print Std Dev Row
            fprintf('%*s %*s (%*.*f) ', colW_m, ' ', colW_d, ' ', ...
                colW_s-2, prec, bma_std(model_idx)); % Adjust width for parentheses
            for j = 1:num_alpha_levels
                fprintf('(%*.*f) ', colW_s-2, prec, lop_stds(model_idx, j));
            end
            fprintf('\n');
        end

        % --- Print Separator ---
        fprintf('%s\n', repmat('-', 1, abs(colW_m) + (2+num_alpha_levels)*(colW_d+1) + colW_d+1 )); % Dynamic width separator

        % --- Print Eta Rows ---
        % Eta_0 Row
        fprintf('%*s %*s %*s ', colW_m, ' ', colW_d, ' ', colW_d, 'Eta_0');
        fprintf('%*.*f ', colW_d, prec, eta0_bma); % Eta0 for BMA (using j=1)
        for j = 1:num_alpha_levels
            fprintf('%*.*f ', colW_d, prec, eta0_vals(j));
        end
        fprintf('\n');

        % Eta_T Row
        fprintf('%*s %*s %*s ', colW_m, ' ', colW_d, ' ', colW_d, 'Eta_T');
         if ~isnan(etaT_bma)
             fprintf('%*.*f ', colW_d, prec, etaT_bma); % EtaT for BMA (using j=1)
         else
              fprintf('%*s ', colW_d, 'N/A');
         end
        for j = 1:num_alpha_levels
             if ~isnan(etaT_vals(j))
                 fprintf('%*.*f ', colW_d, prec, etaT_vals(j));
             else
                 fprintf('%*s ', colW_d, 'N/A');
             end
        end
        fprintf('\n');

        fprintf('--------------------------------------------------------------------------------\n');
        fprintf(' NOTE: Results averaged over %d simulations. Std Dev in parentheses.\n', reps);

    end % k_idx loop (time points)
end % i loop (w* sets)
fprintf('\n--- End Summary Tables ---\n');

%% Display Final Time Step Results from Paths (Unchanged - uses vals)
fprintf('\nDisplaying Final Time Step Results from Paths:\n');
% ... (rest of this display block remains the same as before) ...
for i=1:4
    for j=1:num_alpha_levels % Display paths for j=1,2,3,4
        % ... (code to display final time step results) ...
        if isempty(vals{i,j})
             warning('Skipping display for empty cell vals{%d,%d}', i, j);
             continue;
        end
        total_cols = size(vals{i,j}, 2);
        num_models = size(vals{i,j}, 1);
        if num_models ~= q
             warning('Number of rows in vals{%d,%d} (%d) does not match model count q (%d)', i, j, num_models, q);
             continue;
        end
        num_time_points_path = (total_cols - 1) / 2;
        if floor(num_time_points_path) ~= num_time_points_path || num_time_points_path < 1
            warning('Could not infer consistent number of time points from vals{%d,%d}.', i, j);
            continue;
        end
        T_actual = num_time_points_path - 1;
        col_idx_wstar_start = 1;
        col_idx_bs_start = 1 + 1;
        col_idx_ws_start = 1 + 1 + num_time_points_path;
        col_idx_wstar_1 = col_idx_wstar_start;
        col_idx_bs_final_1 = col_idx_bs_start + num_time_points_path - 1;
        col_idx_ws_final_1 = col_idx_ws_start + num_time_points_path - 1;
        col_idx_display = [col_idx_wstar_1, col_idx_bs_final_1, col_idx_ws_final_1];
         if size(vals{i,j}, 2) < max(col_idx_display) || size(vals{i,j},1) < 1
            warning('Data size insufficient in vals{%d,%d}. Skipping display.', i, j);
            continue;
        end
        Eta_0_disp = q * v(j);
        z=['Simulations = ', num2str(reps), ' Max Time = ',num2str(T_actual)];
        disp(z)
        zz=['Eta_0 = ', num2str(Eta_0_disp), ' (Alpha factor = ', num2str(v(j)), ')'];
        disp(zz)
        zzz=['    Exact    BMA(T)   Est(w)(T)'];
        disp(zzz)
        disp(cell2mat(compose('%10.4f', vals{i,j}(1, col_idx_display ))))
        disp(' ');
    end
end


%% Plot time paths (Example for i=1, Alpha factor=0.5, T=100)
j_to_plot = 2; % Index corresponding to v = [0.1 0.5 1 2]
max_plot_time = 100; % Only plot up to T=100

fprintf('\nGenerating example plot for i=1, j=%d (Alpha factor=%.2f), T=%d...\n', ...
        j_to_plot, v(j_to_plot), max_plot_time);

if num_alpha_levels >= j_to_plot && ~isempty(vals{1,j_to_plot})

    figure(1); clf; % Clear figure window
    hold on;

    % Extract data using j_to_plot
    num_models_plot = size(vals{1,j_to_plot}, 1);
    % T_sim_available = number of time points - 1 (e.g., 1000)
    num_time_points_available = (size(vals{1,j_to_plot}, 2) - 1) / 2;

    plot_time_vector = 0:max_plot_time;
    % Indices needed from the path data (1 to max_plot_time + 1 for t=0 to t=max_plot_time)
    plot_indices = 1:(max_plot_time + 1);

    % Ensure we don't try to plot beyond available data
    if max_plot_time >= num_time_points_available
        warning('Plotting only up to available simulation time T=%d', num_time_points_available - 1);
        plot_indices = 1:num_time_points_available;
        plot_time_vector = 0:(num_time_points_available - 1);
        max_plot_time = num_time_points_available - 1; % Adjust limit for plot commands
    end

    % Extract wstar using j_to_plot
    wstar_1_plot = vals{1,j_to_plot}(1, 1);

    % Calculate column ranges for full paths first
    bs_start_col = 1 + 1;
    bs_end_col = bs_start_col + num_time_points_available - 1;
    ws_start_col = bs_end_col + 1;
    ws_end_col = ws_start_col + num_time_points_available - 1;

    % Get the sliced data for row 1 (Expert 1) using plot_indices
    bs_path_1_plot = vals{1,j_to_plot}(1, bs_start_col -1 + plot_indices);
    ws_path_1_plot = vals{1,j_to_plot}(1, ws_start_col -1 + plot_indices);

    plot(plot_time_vector, bs_path_1_plot,'-k','linewidth',1.5, 'DisplayName', 'E[b_1]');
    plot(plot_time_vector, ws_path_1_plot,'-.k','linewidth',1.5, 'DisplayName', 'E[w_1]');
    plot([plot_time_vector(1) plot_time_vector(end)], [wstar_1_plot wstar_1_plot],':k','linewidth',1.5, 'DisplayName', 'w*_1 (Exact)'); % Line spans plotted range

    hold off;
    ylim([0 1]);
    xlim([0 max_plot_time]); % Set x-axis limit
    xlabel('Time');
    ylabel('Weight Estimate');
    title(sprintf('Expected Time Path (w* set 1, Alpha factor %.2f)', v(j_to_plot)));
    h = legend('show', 'location','eastoutside');
    grid on;

    disp('Plot generated.');
else
    fprintf('Plot skipped (Data for i=1, j=%d is empty or j index is out of bounds).\n', j_to_plot);
end


disp('Script finished.');