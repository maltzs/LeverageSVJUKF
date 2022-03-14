% Leverage stochastic-volatility joint unscented Kalman filter (SV-JUKF):
% Model takes the leverage effect into account. Data taken from Standard
% and Poor (S&P) 500 index returns.
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

N_sim = 1;          % number of simulations
T = 0;              % time span defined by dataset later
jumps = T+1;        % no jumps
N_particles = 0;    % particle filter not used
theta = [];         % data not simulated

% Number of estimated parameters defined by dataset later.
M = 0;

% Initial theta estimates not sampled from uniform distribution.
width = 0;

% Initial estimate covariance
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);

Q_noise = 1e-6*ones(1,M);    % parameter estimate variances
sp = true;                   % data from S&P 500 index
figs = true;                 % produce figures and tables
avg = false;                 % only 1 simulation
ukf = false;                 % no UKF comparison
pf = false;                  % no particle filter comparison

% Runs leverage SV-JUKF.
leverage_SVJUKF_sim(N_sim, T, jumps, N_particles, theta, M, width, ...
    P_corr, Q_noise, sp, figs, avg, ukf, pf);