% Define multiple model weight sets (rows must sum to 1)
%set 1


W105 = [0.33 0.33 0.34;
     0.2721 0.3429 0.3850;
     0.3322 0.3307 0.3371;
     0.3327 0.3321 0.3352];

W110 = [0.33 0.33 0.34;
     0.2317 0.3560 0.4123;
     0.3303 0.3252 0.3445;
     0.3314 0.3295 0.3391];

W115 = [0.33 0.33 0.34;
     0.1837 0.3815 0.4348;
     0.3288 0.3204 0.3508;
     0.3301 0.3272 0.3427];

W120 = [0.33 0.33 0.34;
     0.1442 0.4108 0.4450;
     0.3270 0.3162 0.3568;
     0.3288 0.3252 0.3460];






W405 = [0.00 0.00 1.00;
     0.2318 0.3313 0.4370;
     0.3165 0.3269 0.3566;
     0.3213 0.3297 0.3490];

W410 = [0.00 0.00 1.00;
     0.1334 0.3251 0.5415;
     0.2859 0.3155 0.3986;
     0.2986 0.3240 0.3774];

W415 = [0.00 0.00 1.00;
     0.0766 0.3169 0.6065;
     0.2582 0.3033 0.4385;
     0.2784 0.3175 0.4041];

W420 = [0.00 0.00 1.00;
     0.0436 0.3050 0.6514;
     0.2341 0.2905 0.4754;
     0.2602 0.3109 0.1289];

W430 = [0.00 0.00 1.00;
     0.0137 0.2703 0.7160;
     0.1965 0.2668 0.5367;
     0.2297 0.2975 0.4728];

W450 = [0.00 0.00 1.00;
     0.0014 0.2020 0.7966;
     0.1456 0.2272 0.6273;
     0.1859 0.2730 0.5411];

W4100 = [0.00 0.00 1.00;
     0.0000 0.0984 0.9016;
     0.0897 0.1698 0.7405;
     0.1280 0.2287 0.6434];

W41000 = [0.00 0.00 1.00;
     0.0000 0.0000 1.0000;
     0.0207 0.0650 0.9143;
     0.0320 0.1039 0.8642];



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
W=W115;

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
