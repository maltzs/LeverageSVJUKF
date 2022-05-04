function [L, z_hat] = particle_filter(y, T, N, theta, ...
    sigma_eta_hat, prime)
% PARTICLE_FILTER Runs one iteration of leverage
% stochastic volatility particle filter based on (Mao et al., 2020).
%   Inputs:
%   - y: Vector of log-returns series. Must be length T.
%   - T: Time span of run.
%   - N: Number of particles to use in particle filter.
%   - M: Number of estimated parameters.
%   - theta: Vector of theta values to use in particle filter
%   iteration.
%   - sigma_eta_hat: Standard deviation of state process noise
%   used in particle filter.
%   - prime: Logical value whether phi is transformed to phi
%   prime already or not.
%
%   Ouputs:
%   - L: Likelihood value from particle filter.
%   - z_hat: Vector of residual series of length T.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    sig = @(x) 1./(1+exp(-x));    % sigmoid function

    % Threshold generalized asymmetric stochastic volatility
    % model (Mao et al., 2020).
    f = @(z, alpha, gamma_1, gamma_2) alpha*((z < 0)-0.5)+ ...
        gamma_1*z+gamma_2*(abs(z)-sqrt(2/pi));
    
    beta_0_hat = theta(1);

    % Transforms phi prime back to phi if necessary.
    if prime
        phi_hat = sig(theta(2));
    else
        phi_hat = theta(2);
    end
        
    alpha_hat = theta(3);
    gamma_1_hat = theta(4);
    gamma_2_hat = theta(5);

    L = 0;    % likelihood value

    h_particles = randn(1,N)* ...
        sqrt(sigma_eta_hat^2/(1-phi_hat^2))+ ...
        beta_0_hat/(1-phi_hat);    % particles

    % Residual series is computed via the mean of the particles'
    % state.
    z_hat = zeros(1,T);
    z_hat(1) = mean(exp(h_particles/2))*y(1);

    % Particle filter algorithm.
    for t = 2:T
        % Step 1: Sample next state.
        h_particles_tilde = randn(1,N)*sigma_eta_hat + ...
            beta_0_hat + phi_hat*h_particles + ...
            f(y(t-1)*exp(-h_particles/2), alpha_hat, gamma_1_hat, ...
            gamma_2_hat);

        % Step 2: Compute weights using likelihood function.
        w = normpdf(y(t),0,exp(h_particles_tilde/2));
        L = L-log((1/N)*sum(w));

        % Step 3: Normalize weights and resample particles.
        p = w / sum(w);

        % Augmented steps of resampling to ensure continuous
        % estimator.
        % Step (i): Sort particles in ascending order.
        [h_particles_hat, ind] = sort(h_particles_tilde);
        p_hat = p(ind);
    
        % Step (ii): Sample sorted uniform random variables.
        u = zeros(1,N);
        v = rand(1,N);
        u(N) = v(N)^(1/N);
        for n = N-1:-1:1
            u(n) = u(n+1)*v(n)^(1/n);
        end
    
        % Step (iii): Sample indices.
        pi_hat = zeros(1,N+1);
        pi_hat(1) = p_hat(1)/2;
        pi_hat(N+1) = p_hat(N)/2;
        pi_hat(2:N) = (p_hat(2:N) + p_hat(1:N-1))/2;
    
        s = 0;
        j = 1;
        r = zeros(1,N);
        u_star = zeros(1,N);
        for i = 0:N
            s = s+pi_hat(i+1);
            while j <= N && u(j) <= s
                r(j) = i;
                u_star(j) = (u(j)-(s-pi_hat(i+1)))/pi_hat(i+1);
                j = j+1;
            end
        end
    
        % Step (iv): Resample particles.
        h_particles_hat1 = [h_particles_hat(1) h_particles_hat ...
            h_particles_hat(N)];
        h_particles = ...
            (h_particles_hat1(r+2)-h_particles_hat1(r+1)).* ...
            u_star+h_particles_hat1(r+1);

        % Computes residual series.
        z_hat(t) = mean(exp(h_particles/2))*y(t);
    end
end