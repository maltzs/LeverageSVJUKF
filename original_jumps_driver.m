% Original stochastic volatility joint unscented Kalman filter
% (SV-JUKF) of (Langner, 2022): Model is linear and Gaussian.
% Includes jumps in parameter values.
%
% x(t) = beta_0(t-1)+beta_1(t-1)x(t-1)+beta_2(t-1)eta(t-1)
%     with 0 < beta_1(t) < 1, beta_2(t) > 0 known, eta 
%     independently and identically distributed (iid) N(0,1)
% y(t) = 0.5x(t)+ln|z(t)|    with z iid N(0,1)
%
% Samuel Maltz, Master's thesis at The Cooper Union for the
% Advancement of Science and Art (May 2022)

clear;
clc;
close all;

T = 20000;                         % time span of run
jumps = [5000 10000 15000 T+1];    % indices to jump at

% True beta values. Each row (before transpose) is a set of
% beta_2 values to jump to.
beta = [0.1 0.9 1; 0.3 0.95 1; 0.1 0.9 1; -0.1 0.85 1]';

beta_hat = [0 0.8 1];    % initial beta estimates

% Number of estimated parameters. Excludes beta_2.
M = length(beta_hat)-1;

P_corr = 0.5*eye(M+1);    % initial estimate covariance

% Augmented state process noise covariance.
Q = diag([beta_hat(end)^2 1e-6*ones(1,M)]);

sp = false;    % simulated data

% Runs original SV-JUKF.
original_SVJUKF(T, jumps, M, beta, beta_hat, P_corr, Q, sp);