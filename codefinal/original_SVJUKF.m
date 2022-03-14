function original_SVJUKF(T, jumps, M, beta, beta_hat, P_corr, Q, sp, ...
    vt, lambda)
% ORIGINAL_SVJUKF Runs the orginal Stochastic-volatility joint unscented
% Kalman filter (SV-JUKF) model.
%   Inputs:
%   - T: Time span of run. If sp is true, this will be overwritten with the
%   length of the loaded dataset.
%   - jumps: A vector of indices to jump the true beta values. The indices
%   must be strictly increasing and there must be no more than size(beta,2)
%   valid indices less than or equal to T. Set this to T+1 to disable
%   jumps.
%   - M: Number of estimated parameters.
%   - beta: Matrix of true beta values. Each column represents one set of
%   beta values. The number of columns is the number of sets of beta values
%   that can be jumped to. beta(2,i) must be between 0 and 1 and beta(3,i)
%   must be greater than 0 for all columns i.
%   - beta_hat: Vector of initial estimated beta values. Must have length
%   equal to size(beta,1).
%   - P_corr: Initial covariance of augmented state estimate. Must have
%   size (M+1) x (M+1) and be positive definite.
%   - Q: Augmented state noise covariance. Must have size (M+1) x (M+1) and
%   be symmetric.
%   - sp: Logical value whether to use data from the Standard and Poor
%   (S&P) 500 index or simulated data. False represents simulated data.
%   - vt: Logical value whether too use the variance tracking method to
%   estimate beta_2. Note that this variable will sometimes be used as the 
%   scalar 1.
%   - lambda: Scalar momentum parameter for the variance tracking method.
%   For use only when vt is true.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the Advancement
%   of Science and Art (2022).

    sig = @(x) 1./(1+exp(-x));     % sigmoid function
    invsig = @(x) -log(1./x-1);    % inverse sigmoid function
    
    % Moments of nu.
    mu_nu = -0.635;
    sigma_nu = sqrt(1.234);
    S_nu = -1.536;

    k = 1;    % jumps count

    if sp
        % "spall.mat" is a file with variable "u" which is the returns
        % series of the S&P 500 index from 2000 to 2019. Returns were
        % calculated used the closing prices which were obtained from Yahoo
        % Finance.
        load("spall.mat","u");
        T = length(u);
        y = log(abs(u));    % log-abs-log-return
    else
        x = zeros(1,T);               % log-volatility
        u = zeros(1,T);               % log-return
        y = zeros(1,T);               % log-abs-log-return
        q = randn(1,T);               % log-volatility innovations
        z = randn(1,T);               % log-returns innovations
        a_hat = zeros(M+vt+1,T);      % augmented state vectors over time
        beta_plot = zeros(M+vt,T);    % true beta values over time for plot

        % Initial log-volatility is steady-state mean of the log-volatility
        % process.
        x(1) = beta(1,k)/(1-beta(2,k));

        beta_plot(:,1) = beta(1:M+vt,k);
    end

    % Transformation from beta to theta. Variance tracking uses beta_2.
    theta_hat = [beta_hat(1); invsig(beta_hat(2)); beta_hat(3)];
    if ~vt
        theta_hat(3) = log(theta_hat(3));
    end

    % Initial log-volatility estimate is steady-state mean of the
    % log-volatility process using beta_hat.
    a_hat(1,1) = beta_hat(1)/(1-beta_hat(2));
    a_hat(2:M+vt+1,1) = theta_hat(1:M+vt);

    % Initialization for loop. Initial log-volatility variance estimate is
    % steady state variance of the log-volatility process using beta_hat.
    a_hatcorr = a_hat(1:M+1,1);
    var_x = beta_hat(3)^2/(1-beta_hat(2)^2);

    % Simulates true values.
    if ~sp
        for t = 1:T
            % Jumps at indices specified by jumps.
            if t == jumps(k)
                    k = k+1;
            end
            
            u(t) = exp(0.5*x(t))*z(t);
            y(t) = log(abs(u(t)));
            x(t+1) = [1 x(t) q(t)] * beta(:,k);
    
            beta_plot(:,t) = beta(1:M+vt,k);
        end
    end

    % Actual Kalman algorithm.
    for t = 2:T
        % Expand estimated state and covariance to include state process
        % standard normal noise.
        if M == 3
            a_hatcorr(M+2) = 0;
            P_corr = [P_corr zeros(M+1,1); zeros(1,M+1) 1]; %#ok
        end
        
        [a_hatpred, P_pred] = prediction(M, a_hatcorr, P_corr, Q, 0, ...
            false);    % prediction stage
        [a_hatcorr, P_corr] = ...
            correction(M, [a_hatpred; mu_nu], P_pred, sigma_nu, S_nu, ...
            y(t));    % correction stage
        
        % Variance tracking method to infer beta_2
        if vt
            a = 4*(var(y(1:t))-sigma_nu^2);
            if a > 0
                var_x = a;
            end
            newbeta = sqrt(var_x*(1-sig(a_hatcorr(M+1))^2));
            a_hat(end,t) = lambda*a_hat(end,t-1)+(1-lambda)*newbeta;
            Q(1,1) = a_hat(end,t)^2;
        end
    
        a_hat(1:M+1,t) = a_hatcorr;
    end

    % Transforms theta back to beta.
    a_hat(3,:) = sig(a_hat(3,:));
    if M == 3
        a_hat(4,:) = exp(a_hat(4,:));
    end

    x_hat = a_hat(1,:);
    z_hat = u.*exp(-0.5*x_hat);    % residual series

    statistics_tests(z_hat, true);    % statistics on residual series

    % Parameter estimation over time plots.
    t = 0:T-1;
    figure;
    for i = 1:M+vt
        subplot(M+vt,1,i);
        hold on;
        if ~sp
            plot(t,beta_plot(i,:),'b--');
        end

        plot(t,a_hat(i+1,:));
        hold off;
        xlim([0 T-1]);
        xlabel("Time");
        title("\beta_" + (i-1));
    end
end