%% Set up
reps=100000;
T = 1000; % Max simulation time
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
q = num_models;

% Define potential true model weights
w=cell(1,4);
w{1}=[0.2046; 0.3732; 0.4223];
w{2}=[0.0165; 0.6101; 0.3734];
w{3}=[0.6220; 0.2938; 0.0842];
w{4}=[0.0000; 0.0000; 1.0000];

v = [0.1 0.5 1 2];
num_alpha_levels = length(v); % Will be 4

% Define specific time points for summary
summary_times = [10, 30, 50, 100, 1000];
num_summary_times = length(summary_times);

%% Pre-allocate Results Cell Arrays
num_summary_j_levels = 3; % Corresponds to v = [0.1 0.5 1]
vals_summary = cell(4, num_summary_j_levels, num_summary_times); % i by j by time_index.
vals = cell(4, num_alpha_levels); % i by j. For full paths. (4x4)

%% Combined Simulation and Processing Loop
fprintf('Running combined simulations and processing...\n');
for i = 1:4 % various wstar values
    wstar = w{i};
    if size(wstar, 2) ~= 1; wstar = wstar'; end % Ensure wstar is column qx1
    fprintf(' Processing wstar set %d\n', i);

    for j = 1:num_alpha_levels
        fprintf('  Alpha factor %.2f (j=%d)\n', v(j), j);
        Alpha_0 = ones(1, q) * v(j);
        S_0 = repmat(s, reps, 1);
        b_0 = ones(1, q) * 1/q;
        eta_0 = sum(Alpha_0);
        Afunc = @(S,Alpha) ones(size(S,1),1);

        % --- Run Simulation ---
        fprintf('    Running simulation...\n');
        [S, A, Alpha, Sb, Ab, B] = amdpWsim(P, wstar, T, S_0, Alpha_0, b_0, Afunc, 0);
        actual_T_sim = size(Alpha, 3);

        % --- Process Full Paths ---
        fprintf('    Processing full paths...\n');
        Alphas_path = squeeze(mean(Alpha, 1));
        bs_path = squeeze(mean(B, 1));
        ws_path = normalizeP(Alphas_path);
        vals{i,j} = [wstar bs_path ws_path]; % Store in vals{i,j}

        if j <= num_summary_j_levels % Only for v = [0.1 0.5 1]
            fprintf('    Processing summaries for specific times...\n');
            for k_idx = 1:num_summary_times
                l = summary_times(k_idx);
                time_idx = l; 
                if time_idx < 1 || time_idx > actual_T_sim
                   warning('Summary time point t=%d (index %d) is outside the actual simulation index range [1, %d]. Skipping.', l, time_idx, actual_T_sim);
                   continue; % Skip if this time point doesn't exist in the data
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
                    warning('Eta storage assumes q=3 models. Check vals_summary structure.');
                end
                t_summary = repmat(l, q, 1);

                % Store results using j as index (1, 2, or 3)
                vals_summary{i,j,k_idx} = [t_summary wstar bTl stdbTl wTl stdwTl etas_summary(1:q,:)]; % Store qx7
            end % k_idx loop
        end % if j <= num_summary_j_levels

    end % j loop
end % i loop
fprintf('Finished combined simulations and processing.\n');

%% Display Summary Statistics
fprintf('\nDisplaying Summary Statistics Results:\n');
for i = 1:4
    for j = 1:num_summary_j_levels % Display summaries for j=1,2,3
        for k_idx = 1:num_summary_times
            if isempty(vals_summary{i,j,k_idx})
                continue;
            end
            if size(vals_summary{i,j,k_idx}, 2) ~= 7
                warning('Unexpected data structure in vals_summary{%d,%d,%d}. Skipping display.', i, j, k_idx);
                continue;
            end

            time_val = vals_summary{i,j,k_idx}(1,1);
            % Use v(j) to display the correct alpha factor
            z = sprintf('w* set %d, Alpha factor %.2f, Time %d: Estimates of w', i, v(j), time_val);
            disp(z);

            header = sprintf('%10s %18s %18s', 'Exact', 'BMA (Mean(Std))', 'Est(w) (Mean(Std))');
            disp(header);
            data_to_format = vals_summary{i,j,k_idx}(:, 2:6);
            formatted_output = cell(size(data_to_format, 1), 1);
            for row_idx = 1:size(data_to_format, 1)
                formatted_output{row_idx} = sprintf('%10.4f %8.4f(%7.4f) %8.4f(%7.4f)', ...
                                                   data_to_format(row_idx, 1), data_to_format(row_idx, 2), ...
                                                   data_to_format(row_idx, 3), data_to_format(row_idx, 4), ...
                                                   data_to_format(row_idx, 5));
            end
            disp(strjoin(formatted_output, '\n'));
            disp(' ');
        end
    end
end

%% Display Final Time Step Results from Paths
fprintf('\nDisplaying Final Time Step Results from Paths:\n');
if isempty(vals{1,1})
    error('Cannot display path results: vals{1,1} is empty.');
end
total_cols = size(vals{1,1}, 2);
num_models = size(vals{1,1}, 1);
if num_models ~= q
     warning('Number of rows in vals (%d) does not match model count q (%d)', num_models, q);
end
num_time_points_path = (total_cols - 1) / 2;
if floor(num_time_points_path) ~= num_time_points_path || num_time_points_path < 1
    error('Could not infer consistent number of time points from vals structure.');
end
T_actual = num_time_points_path - 1;
col_idx_wstar_start = 1;
col_idx_bs_start = 1 + 1;
col_idx_ws_start = 1 + 1 + num_time_points_path;
col_idx_wstar_1 = col_idx_wstar_start;
col_idx_bs_final_1 = col_idx_bs_start + num_time_points_path - 1;
col_idx_ws_final_1 = col_idx_ws_start + num_time_points_path - 1;
col_idx_display = [col_idx_wstar_1, col_idx_bs_final_1, col_idx_ws_final_1];

for i=1:4
    for j=1:num_alpha_levels % Display paths for j=1,2,3,4
        if isempty(vals{i,j})
             warning('Skipping display for empty cell vals{%d,%d}', i, j);
             continue;
        end
        if size(vals{i,j}, 2) < max(col_idx_display) || size(vals{i,j},1) < 1
            warning('Data size insufficient in vals{%d,%d}. Skipping display.', i, j);
            continue;
        end

        % Use v(j) to display the correct alpha factor
        Eta_0_disp = q * v(j); % Use q here instead of hardcoded 3
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

%% Plot time paths (Example for i=1, j=1)
fprintf('\nGenerating example plot for i=1, j=1...\n');
if num_alpha_levels >= 1 && ~isempty(vals{1,1}) % Check if j=1 was run
    figure(1); clf;
    hold on;

    num_models_plot = size(vals{1,1}, 1);
    num_time_points_plot = (size(vals{1,1}, 2) - 1) / 2; % T+1
    time_vector = 0:(num_time_points_plot-1); % Assumes time runs 0 to T

    wstar_1_plot = vals{1,1}(1, 1);
    bs_path_1_plot = vals{1,1}(1, (1+1) : (1+num_time_points_plot));
    ws_path_1_plot = vals{1,1}(1, (1+1+num_time_points_plot) : end);

    plot(time_vector, bs_path_1_plot,'-k','linewidth',1.5, 'DisplayName', 'E[b_1]');
    plot(time_vector, ws_path_1_plot,'-.k','linewidth',1.5, 'DisplayName', 'E[w_1]');
    plot([time_vector(1) time_vector(end)], [wstar_1_plot wstar_1_plot],':k','linewidth',1.5, 'DisplayName', 'w*_1 (Exact)');

    ylim([0 1]);
    xlabel('Time');
    ylabel('Weight Estimate');
    % Use v(1) for the title, corresponding to the new j=1 (which is v=0.1)
    title(sprintf('Expected Time Path (w* set 1, Alpha factor %.2f)', v(1)));
    h = legend('show', 'location','eastoutside');
    grid on;
    hold off;
    disp('Plot generated.');
else
    disp('Plot skipped (vals{1,1} is empty or j=1 not run).');
end

disp('Script finished.');