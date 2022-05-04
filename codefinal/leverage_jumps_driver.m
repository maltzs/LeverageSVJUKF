% Leverage stochastic-volatility joint unscented Kalman filter (SV-JUKF):
% Model takes the leverage effect into account. Includes jumps in parameter
% values.
%
% x(t) = beta_0+phix(t-1)+f(epsilon, alpha, gamma_1, gamma_2)
%     +sigma_etaq(t)
% with 0 < phi < 1, f(epsilon, alpha, gamma_1, gamma_2) =
%                       alpha(I(epsilon < 0)-0.5)+gamma_1epsilon+
%                       gamma_2(|epsilon|-sqrt(2/pi)), sigma_eta > 0 fixed,
%                       q iid N(0,1)
% y(t) = 0.5x(t)+nu(t)    with nu iid log(|N(0,1)|)
%
% Samuel Maltz, Master's thesis at The Cooper Union for the Advancement
% of Science and Art (2022).

clear;
clc;
close all;

rng(123);

N_sim = 1;                              % number of simulations
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
M = length(theta)-1;

% Width of uniform distribution to sample initial theta estimates.
width = 0.4;

% Initial estimate covariance
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);

Q_noise = 1e-6*ones(1,M);    % parameter estimate variances
sp = false;                  % simulated data
figs = true;                 % produces figures and tables
avg = false;                 % only 1 simulation
ukf = false;                 % no UKF comparison
pf = false;                  % no particle filter comparison

% Runs leverage SV-JUKF.
leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, M, width, ...
    P_corr, Q_noise, sp, figs, avg, ukf, pf);