% Leverage stochastic-volatility joint unscented Kalman filter (SV-JUKF):
% Model takes the leverage effect into account. Comparison to particle
% filter.
%
% x(t) = mu(1-phi)+phix(t-1)+f(epsilon, alpha, gamma_1, gamma_2)
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

% Sets up random number generator for continuous resampling in particle
% filter.
rs = ...
    RandStream('mt19937ar','Seed',123,'NormalTransform','Inversion');
RandStream.setGlobalStream(rs);

N_sim = 10;          % number of simulations
T = 500;             % time span of run
jumps = T+1;          % no jumps
N_particles = 1e2;    % number of particles used in particle filter

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

% Initial estimate covariance
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);

Q_noise = [1e-6 1e-5 1e-7 1e-7 1e-8];    % parameter estimate variances
sp = false;                              % simulated data

% Do not produce figures and tables for each simulation.
figs = false;

avg = true;     % average autocorrelations of simulations
ukf = false;    % no UKF comparison
pf = true;      % run particle filter comparison

% Runs leverage SV-JUKF.
leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, M, width, ...
    P_corr, Q_noise, sp, figs, avg, ukf, pf);