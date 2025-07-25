% Define multiple model weight sets (rows must sum to 1)
%set 1
W1 = [0.33 0.33 0.34;
     0.0873 0.4610 0.4517;
     0.3224 0.3093 0.3682;
     0.3272 0.3219 0.3509];

W110 = [0.33 0.33 0.34;
     0.2317 0.3560 0.4123;
     0.3304 0.3253 0.3443;
     0.3314 0.3295 0.3391];

%set2
W2 = [0.20 0.19 0.61;
     0.0388 0.3845 0.5767;
     0.2588 0.2955 0.4456;
     0.2796 0.3143 0.4061];

W210 = [0.20 0.19 0.61;
     0.1803 0.3463 0.4734;
     0.3097 0.3215 0.3687;
     0.3166 0.3273 0.3560];


%set3
W3 = [0.01 0.20 0.79;
     0.0207 0.3193 0.6600;
     0.2157 0.2801 0.5041;
     0.2460 0.3060 0.4480];


%set4
W4 = [0.49 0.49 0.02;
     0.2420 0.4872 0.2708;
     0.4286 0.3077 0.2637;
     0.4021 0.3218 0.2761];

% Expert model transition probabilities
P = cell(1, 3);
P{1} = [0.99 0.50 0.20 0.05 0.40 0.40 0.60 0.40;
        0.01 0.50 0.80 0.95 0.60 0.60 0.40 0.60];
P{2} = [0.90 0.60 0.55 0.30 0.20 0.10 0.40 0.25;
        0.10 0.40 0.45 0.70 0.80 0.90 0.60 0.75];
P{3} = [0.80 0.50 0.70 0.40 0.70 0.30 0.50 0.20;
        0.20 0.50 0.30 0.60 0.30 0.70 0.50 0.80];

% Reward matrix: rows = states, columns = actions
R = [0   -5   -5   -5;
     20  15   15   15];

gamma = 0.98;
nStates = 2;
nActions = 4;
actionNames = {'DN', 'FG', 'C', 'N'};

fprintf('Model Weight Sensitivity Analysis\n');
fprintf('----------------------------------\n');

% Value function placeholder — can be zeros or initial guesses
V = [0; 20];  % Assume future value of Low = 0, High = 20 (conservative estimate)
W=W210;

for i = 1:size(W, 1)
    w = W(i, :);

    % Build weighted transition matrix: P_combined(s', s, a)
    P_combined = zeros(nStates, nStates, nActions);
    for a = 1:nActions
        for s = 1:nStates
            idx = (a - 1) * nStates + s;
            for sPrime = 1:nStates
                for k = 1:3
                    P_combined(sPrime, s, a) = P_combined(sPrime, s, a) + w(k) * P{k}(sPrime, idx);
                end
            end
        end
    end

    % Compute policy for each state
    policy = zeros(nStates, 1);
    for s = 1:nStates
        bestVal = -inf;
        for a = 1:nActions
            expectedFuture = sum(P_combined(:, s, a) .* V);
            val = R(s,a) + gamma * expectedFuture;
            if val > bestVal
                bestVal = val;
                policy(s) = a;
            end
        end
    end

    % Display result
    fprintf('Weights: [%.4f %.4f %.4f] -> Policy: Low=%s, High=%s\n', ...
        w(1), w(2), w(3), actionNames{policy(1)}, actionNames{policy(2)});
end
