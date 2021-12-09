clear; close all; clc;

rs = ...
    RandStream('mt19937ar','Seed','shuffle','NormalTransform','Inversion');
RandStream.setGlobalStream(rs);

f = @(epsilon, alpha, gamma_1, gamma_2) alpha*((epsilon < 0) - 0.5) ...
        + gamma_1*epsilon + gamma_2*(abs(epsilon) - sqrt(2/pi));

T = 10000;
N = 1e4;
R = 100;
pv_trials = 1;
ukf_iterations = 1;

% True parameters
mu = 0;
phi = 0.8;
alpha = 0.07;
gamma_1 = -0.08;
gamma_2 = 0.1;
sigma_eta = sqrt(0.05);

M = 5;

theta_pv = zeros(R,M);
theta_ukf = zeros(R,M);

for i = 1:R
    fprintf("Simulation " + i + "\n");

    % Approx parameters
    width = 0.6;
    mu_hat = mu - width/2 + width*rand;
%     phi_hat = phi - width/2 + width*rand;
    phi_hat = phi - width/2 + (1-phi+width/2)*rand;
    alpha_hat = alpha - width/2 + width*rand;
    gamma_1_hat = gamma_1 - width/2 + width*rand;
    gamma_2_hat = gamma_2 - width/2 + width*rand;
%     sigma_eta_hat = 0 + (sigma_eta+width/2)*rand;

    theta_hat = [mu_hat*(1 - phi_hat) phi_hat alpha_hat gamma_1_hat ...
        gamma_2_hat];

    h = zeros(1,T+1);
    u = zeros(1,T);
    y = zeros(1,T);
    
    h(1) = randn*sqrt(sigma_eta^2/(1-phi^2)) + mu;
    for t = 1:T
        epsilon = randn;
        u(t) = exp(h(t)/2)*epsilon;
        y(t) = log(abs(u(t)));
        h(t+1) = mu*(1-phi) + phi*h(t) ...
            + f(epsilon, alpha, gamma_1, gamma_2) + randn*sigma_eta;
    end

%     theta_pv(i,:) = run_pv_sim(T, N, pv_trials, u, theta_hat);
%     if all(~isnan(theta_pv(i,:)))
%         [~, z_hat] = particle_filter(u, T, N, theta_pv(i,:), true);
%         stats_pv(i) = statistics_tests(u, z_hat);   %#ok
%     end

    theta_ukf(i,:) = run_ukf_sim(T, ukf_iterations, u, y, theta_hat);
    if all(~isnan(theta_ukf(i,:)))
        z_hat = ukf_pure(T,u,y,theta_ukf(i,:));
        stats_ukf(i) = statistics_tests(u, z_hat);  %#ok
    end
end

% theta_pv_avg = mean(theta_pv,'omitnan');
% theta_pv_std = std(theta_pv,'omitnan');

theta_ukf_avg = mean(theta_ukf,'omitnan');
theta_ukf_std = std(theta_ukf,'omitnan');