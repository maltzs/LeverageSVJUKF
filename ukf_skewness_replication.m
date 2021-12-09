function ukf_skewness_replication(x_0, x_hat0, b, beta_hat, P_corr)
    T = 100;
    
    x = zeros(1,T);
    y = zeros(1,T);
    q = randn(1,T);
    v = log(abs(randn(1,T)));
    
    mu_v = -0.635;
    sigma_v = sqrt(1.234);
    S_v = -1.536;
    
    x(1) = x_0;                           % intial actual x value
    c = 0.5;                            % alpha in D matrix
    M = length(b) - 1;                  % excludes beta_2 for now
    Q = diag([b(end)^2 zeros(1,M)]);    % noise matrix
    
    a_hat = zeros(M+1,T);   % augmented state vectors over time
    y_hat = zeros(1,T);     % measurements over time
    
    a_hat(1,1) = x_hat0;      % initial x approx value
    a_hat(2:M+1,1) = [beta_hat(1); -log(1/beta_hat(2) - 1)];    % theta_hat
    
    sig = @(x) 1./(1+exp(-x));
    
    % Initialization for loop
    a_hatcorr = a_hat(:,1);
    
    for t = 2:T
        % Actual values
        x(t) = [1 x(t-1) q(t)] * b;
        y(t) = c*x(t) + v(t);
        
        [a_hatpred, P_pred, flag] = prediction(M, a_hatcorr, P_corr, Q);
        
        if flag > 0
            break
        end

        [a_hatcorr, P_corr, y_hat(t), flag] = ...
            correction(M, c, [a_hatpred; mu_v], P_pred, sigma_v, S_v, y(t));
        
        if flag > 0
            break
        end

        a_hat(:,t) = a_hatcorr;
    end
    
    s = t;
    t = 0:T-1;
    figure;
    subplot(2,1,1);
    plot(t,x,'b--',t,a_hat(1,:));
    xlim([0 T-1]);
    title("x (t = " + s + ",x_0 = " + x_0 + ")");
    
    subplot(2,1,2);
    plot(t,y,'b--',t,y_hat);
    xlim([0 T-1]);
    title("y");
    
    a_hat(3,:) = sig(a_hat(3,:));
    
%     figure;
%     for i = 1:M
%         subplot(M,1,i);
%         plot(t,b(i)*ones(1,T),'b--',t,a_hat(1+i,:));
%         xlim([0 T-1]);
%         title("\beta_" + i);
%     end
end
    
function [a_hatpred, P_pred, flag] = prediction(M, mu, P, Q)
    sig = @(x) 1./(1+exp(-x));

    [w, chi, flag] = create_gaussian_sigma_points(mu, P);

    if flag > 0
        a_hatpred = [];
        P_pred = [];
        return
    end
    
    % Pass sigma points through process equation
    chi_x = chi(1,:);
    chi_theta = chi(2:M+1,:);
    chi_beta = [chi_theta(1,:); sig(chi_theta(2,:))];
    chi_pred = [sum(chi_beta .* [ones(1,length(chi_x)); chi_x]); ...
        chi_theta];
    
    % Compute predicted augmented state and covariance matrix
    a_hatpred = chi_pred * w;
    P_pred = ...
        (chi_pred - a_hatpred) * diag(w) * (chi_pred - a_hatpred)' + Q;
end

function [a_hatcorr, P_corr, y_hat, flag] = ...
    correction(M, c, mu, P, sigma_v, S_v, y)
    [w, chi, flag] = create_sigma_points(M, c, mu, P, sigma_v, S_v);
    
    if flag > 0
        a_hatcorr = [];
        P_corr = [];
        y_hat = 0;
        return
    end

    % Compute empirical correlation matrices
    chi_x = chi(1,:);
    chi_theta = chi(2:M+1,:);
    chi_y = chi(M+2,:);
    chi_alpha = y - chi_y;
    
    R_xalpha = sum(w' .* chi_x .* chi_alpha);
    R_thetaalpha = sum(w' .* chi_theta .* chi_alpha,2);
    R_alphaalpha = sum(w' .* chi_alpha .* chi_alpha);
    
    chi_corr = [chi_x + (1/R_alphaalpha)*R_xalpha * chi_alpha; ...
        chi_theta + (1/R_alphaalpha)*R_thetaalpha .* chi_alpha];
    
    % Compute corrected augmented state and covariance matrix
    a_hatcorr = chi_corr * w;
    P_corr = (chi_corr - a_hatcorr) * diag(w) * (chi_corr - a_hatcorr)';

%     alpha_hat = chi_alpha * w;
%     R_aalpha = [R_xalpha; R_thetaalpha];
%     
%     a_hatcorr = mu(1:end-1) + R_aalpha * (1/R_alphaalpha) * alpha_hat;
%     P_corr = P - R_aalpha * R_aalpha' * (1/R_alphaalpha);
    y_hat = chi_y * w;
end

function [w, chi, flag] = create_gaussian_sigma_points(mu, C)
    M = length(mu);
    [L, flag] = chol(C,'lower');

    if flag > 0
        chi = [];
        w = [];
        return
    end

    chi = mu + sqrt(M) * [L -L];
    w = (1/(2*M)) * ones(2*M,1);
end

function [w, chi, flag] = create_sigma_points(M, c, mu, C, sigma_v, S_v)
    a = 1/(M+2);

    % Gaussian components
    s = sqrt(M+2) * ones(1,2*(M+2));
    w = (1/(2*(M+2))) * ones(2*(M+2),1);
    
    % Non-gaussian components
    s(M+2) = 0.5*(-S_v + sqrt(a^2*S_v^2 + 4/a));
    s(2*(M+2)) = 0.5*(S_v + sqrt(a^2*S_v^2 + 4/a));
    w(M+2) = a*s(2*(M+2))/(s(M+2)+s(2*(M+2)));
    w(2*(M+2)) = a*s(M+2)/(s(M+2)+s(2*(M+2)));
    
    chi_0 = [diag(s(1:M+2)) -diag(s(M+3:end))];
    
    Dinv = eye(M+2);
    Dinv(end,1) = c;
    [L, flag] = chol(C,'lower');

    if flag > 0
        chi = [];
        return
    end
    
    A = zeros(M+2);
    A(1:M+1,1:M+1) = L;
    A(end,end) = sigma_v;
    
    chi = Dinv * (A * chi_0 + mu);
end