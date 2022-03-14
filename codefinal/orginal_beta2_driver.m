% Original stochastic-volatility joint unscented Kalman filter (SV-JUKF):
% Model is linear and Gaussian.
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

T = 20000;                            % time span of run
jumps = T+1;                          % no jumps
beta = [0.1; 0.9; 1];                 % true beta values
beta_hat = [0.2 0.8 0.8];             % initial beta estimates
M = length(beta_hat);                 % number of estimated parameters
P_corr = 0.1*ones(4) + 0.4*eye(4);    % initial estimate covariance

% Augmented state process noise covariance not used when beta_2 is
% estimated using the SV-JUKF.
Q = 0;

sp = false;    % simulated data

% beta_2 estimated using SV-JUKF.
vt = false;
lambda = 0;

% Runs original SV-JUKF.
original_SVJUKF(T, jumps, M, beta, beta_hat, P_corr, Q, sp, vt, lambda);