% Original stochastic volatility joint unscented Kalman filter
% (SV-JUKF) of (Langner, 2022): Model is linear and Gaussian.
% Data taken from Standard and Poor (S&P) 500 index returns.
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

% Time span defined by dataset later.
T = 0;

jumps = T+1;                        % no jumps
beta = [];                          % data not simulated
beta_hat = [0.2 0.9 sqrt(0.36)];    % initial beta estimates

% Number of estimated parameters. Excludes beta_2.
M = length(beta_hat)-1;

P_corr = 0.5*eye(M+1);    % initial estimate covariance

% Augmented state process noise covariance.
Q = diag([beta_hat(end)^2 1e-6*ones(1,M)]);

sp = true;    % Data from S&P 500 index

% Runs original SV-JUKF.
original_SVJUKF(T, jumps, M, beta, beta_hat, P_corr, Q, sp);