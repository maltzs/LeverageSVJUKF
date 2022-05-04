% Leverage stochastic volatility joint unscented Kalman filter
% (SV-JUKF): Model takes the leverage effect into account.
% Comparison to particle filter based on (Mao et al., 2020).
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

% Sets up random number generator for continuous resampling in
% particle filter.
rs = RandStream('mt19937ar','Seed','shuffle','NormalTransform', ...
    'Inversion');
RandStream.setGlobalStream(rs);

N_sim = 100;          % number of simulations
T = 2000;             % time span of run
jumps = T+1;          % no jumps
N_particles = 1e4;    % number of particles used in particle filter

% True theta values.
mu = 0;
phi = 0.98;
alpha = 0.07;
gamma_1 = -0.08;
gamma_2 = 0.1;
sigma_eta = sqrt(0.05);
theta = [mu*(1-phi); phi; alpha; gamma_1; gamma_2; sigma_eta];

% Number of estimated parameters. Excludes sigma_eta.
M = length(theta)-1;

% Width of uniform distribution to sample initial theta estimates.
width = 0.6;

% Initial estimate covariance.
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);

% Parameter estimate variances.
Q_noise = [1e-6 1e-5 1e-7 1e-7 1e-8];

sp = false;    % simulated data

% Do not produce figures and tables for each simulation.
figs = false;

avg = true;     % average autocorrelations of simulations
ukf = false;    % no UKF comparison
pf = true;      % run particle filter comparison

% Runs leverage SV-JUKF.
leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, M, ...
    width, P_corr, Q_noise, sp, figs, avg, ukf, pf);