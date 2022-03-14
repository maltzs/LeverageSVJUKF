% Original stochastic-volatility joint unscented Kalman filter (SV-JUKF):
% Model is linear and Gaussian. beta_2 estimated using variance tracking
% method.
%
% x(t) = beta_0+beta_1x(t-1)+beta_2q(t)
%                             with 0 < beta_1 < 1, beta_2 > 0, q iid N(0,1)
% y(t) = 0.5x(t)+nu(t)    with v iid log(abs(N(0,1)))
%
% Samuel Maltz, Master's thesis at The Cooper Union for the Advancement
% of Science and Art (2022).

clear;
clc;
close all;

rng(123);

T = 20000;    % time span of run
jumps = T+1;    % no jumps
beta = [0.1; 0.9; 1];    % true beta values
beta_hat = [0.2 0.8 0.8];    % initial beta estimates

% Number of parameters estimated with the SV-JUKF directly. Excludes beta_2.
M = length(beta_hat)-1;

P_corr = 0.5*eye(M+1);    % initial estimate covariance

% Augmented state process noise covariance.
Q = diag([beta_hat(end)^2 1e-6*ones(1,M)]);

sp = false;      % simulated data
vt = true;       % variance tracking method
lambda = 0.9;    % momentum parameter

% Runs original SV-JUKF.
original_SVJUKF(T, jumps, M, beta, beta_hat, P_corr, Q, sp, vt, lambda);