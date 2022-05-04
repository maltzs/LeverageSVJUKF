% Leverage stochastic volatility joint unscented Kalman filter
% (SV-JUKF): Model takes the leverage effect into account.
% Comparison to standard UKF.
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

N_sim = 100;                              % number of simulations
T = 100000;                               % time span of run
jumps = [20000 40000 60000 80000 T+1];    % indices to jump at
N_particles = 0;                          % particle filter not used

% True theta values. Each entry is a value to jump to.
mu = [0 1 1 1 1];
phi = [0.8 0.8 0.98 0.98 0.98];
alpha = [0.07 0.07 0.07 0.27 0.27];
gamma_1 = [-0.08 -0.08 -0.08 -0.08 -0.28];
gamma_2 = [0.1 0.1 0.1 0.1 0.1];
sigma_eta = [sqrt(0.05) sqrt(0.05) sqrt(0.05) sqrt(0.05) sqrt(0.05)];
theta = [mu.*(1-phi); phi; alpha; gamma_1; gamma_2; sigma_eta];

% Number of estimated parameters. Excludes sigma_eta.
M = length(theta(:,1))-1;

% Width of uniform distribution to sample initial theta estimates.
width = 0.4;

% Initial estimate covariance.
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);

% Parameter estimate variances.
Q_noise = [1e-6 1e-5 1e-7 1e-7 1e-8];

sp = false;    % simulated data

% Do not produce figures and tables for each simulation.
figs = false;

avg = true;    % average autocorrelations of simulations
ukf = true;    % run UKF comparison
pf = false;    % no particle filter comparison

% Runs leverage SV-JUKF.
leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, M, ...
    width, P_corr, Q_noise, sp, figs, avg, ukf, pf);