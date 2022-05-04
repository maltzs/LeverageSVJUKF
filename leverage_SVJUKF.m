function [a_hat, z_hat] = leverage_SVJUKF(T, M, theta_hat, ...
    P_corr, Q, u, y)
% LEVERAGE_SVJUKF Runs the leverage stochastic volatility joint
% unscented Kalman filter (SV-JUKF) model.
%   Inputs:
%   - T: Time span of run.
%   - M: Number of estimated parameters.
%   - theta_hat: Vector of initial estimated theta values.
%   - P_corr: Initial covariance of augmented state estimate.
%   Must have size (M+1)x(M+1) and be positive definite.
%   - Q: Augmented state noise covariance. Must have size
%   (M+1)x(M+1) and be symmetric.
%   - u: Vector of log-returns series. Must be length T.
%   - y: Vector of log-abs-log-returns series. Must be length T.
%
%   Outputs:
%   - a_hat: Matrix of estimated augmented states over times of
%   size (M+1)xT.
%   - z_hat: Vector of residual series of length T.
%
%   Throws:
%   - MATLAB:posdec: From prediction or correction if P_pred or
%   P_corr is not positive definite.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    sig = @(x) 1./(1+exp(-x));      % sigmoid function
    inv_sig = @(x) -log(1./x-1);    % inverse sigmoid function
    
    % Moments of nu = ln|z|, z ~ N(0,1).
    mu_nu = -0.635;
    sigma_nu = sqrt(1.234);
    S_nu = -1.535;

    a_hat = zeros(M+1,T);    % augmented state vectors over time  
    
    % Initial log-volatility estimate is steady-state mean of the
    % log-volatility process using theta_hat.
    a_hat(1,1) = theta_hat(1)/(1-theta_hat(2));
    if M > 0
        % Transforms phi to phi prime.
        a_hat(2:M+1,1) = [theta_hat(1); inv_sig(theta_hat(2)); 
            theta_hat(3); theta_hat(4); theta_hat(5)];
    end
    
    % Initialization for loop.
    a_hat_corr = a_hat(:,1); 

    % Actual Kalman algorithm.
    for t = 2:T
        % Previous time step's residual.
        z = u(t-1)*exp(-0.5*a_hat(1,t-1));
    
        [a_hat_pred, P_pred] = prediction(M, a_hat_corr, P_corr, ...
            Q, z, true, theta_hat);           % prediction stage
        [a_hat_corr, P_corr] = correction(M, [a_hat_pred; mu_nu], ...
            P_pred, sigma_nu, S_nu, y(t));    % correction stage
    
        a_hat(:,t) = a_hat_corr;
    end
    
    % Transforms phi prime back to phi.
    if M > 0
        a_hat(3,:) = sig(a_hat(3,:));
    end
    
    x_hat = a_hat(1,:);
    z_hat = u .* exp(-0.5*x_hat);    % residual series
end