function leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, ...
    M, width, P_corr, Q_noise, sp, figs, avg, ukf, pf)
% LEVERAGE_SVJUKF_SIM Wrapper for the leverage
% stochastic volatility joint unscented Kalman filter (SV-JUKF)
% model. Runs a certain number of simulations.
%   Inputs:
%   - N_sim: Number of simulations to perform.
%   - T: Time span of run. If sp is true, this will be
%   overwritten with the length of the loaded dataset.
%   - jumps: A vector of indices to jump the true beta values.
%   The indices must be strictly increasing and there must be no
%   more than size(beta,2) valid indices less than or equal to T.
%   Set this to T+1 to disable jumps.
%   - N_particles: Number of particles to use in particle filter.
%   For use only when pf is true.
%   - theta: Matrix of true theta values. Each column represents
%   one set of theta values. The number of columns is the number
%   of sets of theta values that can be jumped to. theta(2,i) 
%   must be between 0 and 1 and theta(3,i) must be greater than 0
%   for all columns i.
%   - M: Number of estimated parameters.
%   - width: Width of uniform distribution to sample initial theta
%   estimates from.
%   - P_corr: Initial covariance of augmented state estimate.
%   Must have size (M+1)x(M+1) and be positive definite.
%   - Q_noise: Vector of parameter estimate variances. Must have
%   length M.
%   - sp: Logical value whether to use data from the Standard and
%   Poor (S&P) 500 index or simulated data. False represents
%   simulated data.
%   - figs: Logical value whether figures and tables should be
%   produced for every simulation or not.
%   - avg: Logical value whether to average and plot the absolute
%   value sample autocorrelations of the simulations or not.
%   - ukf: Logical value whether to also run a UKF to compare
%   against the JUKF or not.
%   - pf: Logical value whether to also run a particle filter to
%   compare against the JUKF or not.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    % Threshold generalized asymmetric stochastic volatility
    % model (Mao et al., 2020).
    f = @(z, alpha, gamma_1, gamma_2) alpha*((z < 0)-0.5)+ ...
        gamma_1*z+gamma_2*(abs(z)-sqrt(2/pi));

    % Used to compare parameter estimates and elapsed time for
    % the particle filter and SV-JUKF.
    if pf
        theta_jukf = zeros(N_sim,M);
        theta_pf = zeros(N_sim,M);

        time_jukf = zeros(N_sim,1);
        time_pf = zeros(N_sim,1);
    end

    n = 1;
    while n <= N_sim
        k = 1;    % jumps count

        if sp
            % Initial theta estimates.
            theta_hat = [0 0.9 0.07 -0.08 0.1 sqrt(0.16)];

            % Number of estimated parameters. sigma_eta is excluded.
            M = length(theta_hat)-1;

            % "sp500.mat" is a file with variable "u" which is
            % the returns series of the S&P 500 index from 2000
            % to 2019. % Returns were calculated using the
            % closing prices obtained from the Wall Street
            % Journal at https://www.wsj.com/market-data/quotes/
            % index/SPX/historical-prices.
            load("sp500.mat","u");
            T = length(u);

            y = log(abs(u));    % log-abs-log returns
        else
            % Initial theta estimates. phi_hat is ensured to be
            % between 0 and 1.
            mu_hat = theta(1,k)-width/2+width*rand;
            phi_hat = theta(2,k)-width/2+(1-theta(2,k)+width/2)*rand;
            alpha_hat = theta(3,k)-width/2+width*rand;
            gamma_1_hat = theta(4,k)-width/2+width*rand;
            gamma_2_hat = theta(5,k)-width/2+width*rand;
            sigma_eta_hat = theta(6,k);
            theta_hat = [mu_hat*(1-phi_hat) phi_hat alpha_hat ...
                gamma_1_hat gamma_2_hat sigma_eta_hat];

            x = zeros(1,T+1);    % log-volatility
            u = zeros(1,T);      % log-returns
            y = zeros(1,T);      % log-abs-log-returns

            % True theta values over time for plots.
            theta_plot = zeros(M,T); 
            
            % Initial log-volatility is steady-state mean of the
            % log-volatility process.
            x(1) = theta(1,k)/(1-theta(2,k));
        end
        
        % Simulates true values.
        if ~sp
            for t = 1:T
                % Jumps at indices specified by jumps.
                if t == jumps(k)
                    k = k+1;
                end
        
                z = randn;
                u(t) = exp(0.5*x(t))*z;
                y(t) = log(abs(u(t)));
                x(t+1) = theta(1,k)+theta(2,k)*x(t)+f(z, ...
                    theta(3,k), theta(4,k), ...
                    theta(5,k))+randn*theta(6,k);
        
                theta_plot(:,t) = [theta(1,k) theta(2,k) ...
                    theta(3,k) theta(4,k) theta(5,k)];
            end
        end

        % Augmented state process noise covariance.
        Q = diag([theta_hat(6)^2 Q_noise]);

        try
            % Runs leverage SV-JUKF. On off chance of model 
            % breakdown, redoes simulation (see catch).
            tic;
            [a_hat_jukf, z_hat_jukf] = leverage_SVJUKF(T, M, ...
                theta_hat, P_corr, Q, u, y);
            time_jukf(n) = toc;
            if any(isnan(a_hat_jukf) | isinf(a_hat_jukf),'all') ...
                || any(isnan(z_hat_jukf) | isinf(z_hat_jukf))
                throw(MException('MATLAB:nonaninf',"NaN or Inf"));
            end
            
            if ukf
                % Runs leverage SV-UKF. On off chance of model
                % breakdown, redoes simulation (see catch).
                [a_hat_ukf, z_hat_ukf] = leverage_SVJUKF(T, 0, ...
                    theta_hat, 0.5, 0.05, u, y);
                if any(isnan(a_hat_ukf) | isinf(a_hat_ukf)) || ...
                        any(isnan(z_hat_ukf) | isinf(z_hat_ukf))
                    throw(MException('MATLAB:nonaninf', ...
                        "NaN or Inf"));
                end
            end

            if pf
                % Runs leverage SV-UKF with final theta estimates
                % from SV-JUKF to compute residual series to
                % compare with particle filter. On off chance of
                % model breakdown, redoes simulation (see catch).
                theta_jukf(n,:) = a_hat_jukf(2:end,end)';
                [~, z_hat_ukf] = leverage_SVJUKF(T, 0, ...
                    theta_jukf(n,:), 0.5, 0.05, u, y);
                if any(isnan(z_hat_ukf) | isinf(z_hat_ukf))
                    throw(MException('MATLAB:nonaninf', ...
                        "NaN or Inf"));
                end

                % Runs particle filter to estimate theta. Then reruns
                % particle filter once to compute residual series to
                % compare with SV-JUKF. On off chance of model
                % breakdown, redoes simulation (see catch).
                tic;
                theta_pf(n,:) = leverage_SVPF(T, N_particles, M, ...
                    u, theta_hat);
                time_pf(n) = toc;
                [~, z_hat_pf] = particle_filter(u, T, ...
                    N_particles, theta_pf(n,:), theta_hat(6), false);
                if any(isnan(z_hat_pf) | isinf(z_hat_pf))
                    throw(MException('MATLAB:nonaninf', ...
                        "NaN or Inf"));
                end
            end
        catch ME
            switch ME.identifier
                case 'MATLAB:posdef'
                    continue;
                case 'MATLAB:nonaninf'
                    continue;
                otherwise
                    rethrow(ME);
            end
        end
        
        % Statistics on residual series.
        stats_jukf(n) = statistics_tests(z_hat_jukf, figs);    %#ok
        
        if ukf || pf
            stats_ukf(n) = statistics_tests(z_hat_ukf, figs);    %#ok
        end

        if pf
            stats_pf(n) = statistics_tests(z_hat_pf, figs);    %#ok
        end

        % Parameter estimation over time plots.
        if figs
            t = 0:T-1;
            standard_titles = ["\beta_0","\phi"];
            leverage_titles = ["\alpha","\gamma_1","\gamma_2"];
            figure;
            for i = 1:2
                subplot(2,1,i);
                hold on;
                if ~sp
                    plot(t,theta_plot(i,:),'b--');
                end
        
                plot(t,a_hat_jukf(i+1,:));
                hold off;
                xlim([0 T-1]);
                title(standard_titles(i));
                xlabel("Time");
            end
        
            figure;
            for i = 1:3
                subplot(3,1,i);
                hold on;
                if ~sp
                    plot(t,theta_plot(i+2,:),'b--');
                end
                
                plot(t,a_hat_jukf(i+3,:));
                hold off;
                xlim([0 T-1]);
                title(leverage_titles(i));
                xlabel("Time");
            end
        end

        n = n+1;
    end

    % Averages and plots the absolute value sample
    % autocorrelations of the simulations.
    if avg
        if ~pf
            acf_avg(T, stats_jukf);
        end

        if ukf || pf
            acf_avg(T, stats_ukf);
        end

        if pf
            acf_avg(T, stats_pf);

            table(["\beta_0"; "phi"; "alpha"; "gamma_1"; ...
                "gamma_2"], mean(theta_jukf(:,1:M),1)', ...
                std(theta_jukf(:,1:M),0,1)', ...
                mean(theta_pf(:,1:M),1)', ...
                std(theta_pf(:,1:M),0,1)','VariableNames', ...
                ["Parameter","SV-JUKF avg","SV-JUKF std", ...
                "SV-PF avg","SV-PF std"])

            table("Time",mean(time_jukf,1),std(time_jukf,0,1), ...
                mean(time_pf,1),std(time_pf,0,1),'VariableNames', ...
                ["Quantity","SV-JUKF avg","SV-JUKF std", ...
                "SV-PF avg","SV-PF std"])
        end
    end   
end