% Leverage stochastic volatility joint unscented Kalman filter
% (SV-JUKF): Model takes the leverage effect into account. Data
% taken from Standard and Poor (S&P) 500 index returns.
%
% x(t) = beta_0(t-1)+phi(t-1)x(t-1)+f(z(t-1), alpha(t-1),
%            gamma_1(t-1), gamma_2(t-1))+sigma_eta(t-1)eta(t-1)    
%     with 0 < phi(t) < 1,
%     f(z, alpha, gamma_1, gamma_2) = alpha(I(z < 0)-0.5)+gamma_1z+
%                                         gamma_2(|z|-sqrt(2/pi)),
%     z defined below, sigma_eta(t) > 0 known and eta
%     independently and identically distributed (iid) N(0,1) (Mao
%     et al., 2020)
% y(t) = 0.5x(t)+ln|z(t)|    with z iid N(0,1)
%
% Samuel Maltz, Master's thesis at The Cooper Union for the
% Advancement of Science and Art (May 2022)

clear;
clc;
close all;

N_sim = 1;          % number of simulations
T = 0;              % time span defined by dataset later
jumps = T+1;        % no jumps
N_particles = 0;    % particle filter not used
theta = [];         % data not simulated

% Number of estimated parameters defined by dataset later.
M = 0;

% Initial theta estimates not sampled from uniform distribution.
width = 0;

% Initial estimate covariance.
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);

% Parameter estimate variances.
Q_noise = [1e-6 1e-5 1e-7 1e-7 1e-8];

sp = true;      % data from S&P 500 index
figs = true;    % produce figures and tables
avg = false;    % only 1 simulation
ukf = false;    % no UKF comparison
pf = false;     % no particle filter comparison

% Runs leverage SV-JUKF.
leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, M, ...
    width, P_corr, Q_noise, sp, figs, avg, ukf, pf);