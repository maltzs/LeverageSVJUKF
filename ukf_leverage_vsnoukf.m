clear; close all; clc;

rng(12345);

f = @(epsilon, alpha, gamma_1, gamma_2) alpha*((epsilon < 0) - 0.5) ...
        + gamma_1*epsilon + gamma_2*(abs(epsilon) - sqrt(2/pi));

T = 100000;
R = 100;
ukf_iterations = 1;
start = 1;

jump = [20000 40000 60000 80000 T+1];

% True parameters
mu = [1 1 1 1 1];
phi = [0.8 0.98 0.98 0.98 0.98];
alpha = [0.07 0.07 0.07 0.27 0.27];
gamma_1 = [-0.08 -0.08 -0.08 -0.08 -0.28];
gamma_2 = [0.1 0.1 0.1 0.1 0.1];
sigma_eta = sqrt(0.05);

M = 5;

theta_ukf = zeros(R,M);

for i = 1:R
    fprintf("Simulation " + i + "\n");
    k = 1;

    % Approx parameters
    width = 0.4;
    mu_hat = mu(k) - width/2 + width*rand;
%     phi_hat = phi(k) - width/2 + width*rand;
    phi_hat = phi(k) - width/2 + (1-phi(k)+width/2)*rand;
    alpha_hat = alpha(k) - width/2 + width*rand;
    gamma_1_hat = gamma_1(k) - width/2 + width*rand;
    gamma_2_hat = gamma_2(k) - width/2 + width*rand;
%     sigma_eta_hat = 0 + (sigma_eta+width/2)*rand;

%     % Approx parameters manual
%     mu_hat = 0.2;
%     phi_hat = 0.6;
%     alpha_hat = 0.27;
%     gamma_1_hat = -0.28;
%     gamma_2_hat = 0.3;

    theta_hat = [mu_hat*(1 - phi_hat) phi_hat alpha_hat gamma_1_hat ...
        gamma_2_hat];

    h = zeros(1,T+1);
    u = zeros(1,T);
    y = zeros(1,T);
    
    h(1) = randn*sqrt(sigma_eta^2/(1-phi(k)^2)) + mu(k);
    for t = 1:T
        if t == jump(k)
            k = k + 1;
        end

        epsilon = randn;
        u(t) = exp(h(t)/2)*epsilon;
        y(t) = log(abs(u(t)));
        h(t+1) = mu(k)*(1-phi(k)) + phi(k)*h(t) ...
            + f(epsilon, alpha(k), gamma_1(k), gamma_2(k)) + randn*sigma_eta;
    end

    [theta_ukf(i,:), zhat] = ...
        run_ukf_sim(T, ukf_iterations, u, y, theta_hat, start);
    stats_jukf(i) = statistics_tests(u, zhat);  %#ok
    zhat_nojukf = ukf_pure(T,u,y,theta_hat);
    stats_nojukf(i) = statistics_tests(u, zhat_nojukf);  %#ok
end

theta_ukf_avg = mean(theta_ukf,'omitnan');
theta_ukf_std = std(theta_ukf,'omitnan');

acfavg

fprintf("With JUKF\n")
fprintf("archtest of z_hat: " + stats_jukf(1).arch.archz + " (pvalue: " + stats_jukf(1).arch.parchz + ")\n")
fprintf("LB Q test: " + stats_jukf(1).lbq.lbq + " (pvalue: " + stats_jukf(1).lbq.plbq + ")\n")
fprintf("Mean: " + stats_jukf(1).moments.meanz + "\n")
fprintf("Variance: " + stats_jukf(1).moments.varz + "\n")
fprintf("Skewness: " + stats_jukf(1).moments.skewz + "\n")
fprintf("Kurtosis: " + stats_jukf(1).moments.kurtz + "\n")

fprintf("\nWithout JUKF\n")
fprintf("archtest of z_hat: " + stats_nojukf(1).arch.archz + " (pvalue: " + stats_nojukf(1).arch.parchz + ")\n")
fprintf("LB Q test: " + stats_nojukf(1).lbq.lbq + " (pvalue: " + stats_nojukf(1).lbq.plbq + ")\n")
fprintf("Mean: " + stats_nojukf(1).moments.meanz + "\n")
fprintf("Variance: " + stats_nojukf(1).moments.varz + "\n")
fprintf("Skewness: " + stats_nojukf(1).moments.skewz + "\n")
fprintf("Kurtosis: " + stats_nojukf(1).moments.kurtz + "\n")

figure;
qqplot(zhat);

figure;
qqplot(zhat_nojukf);

figure;
subplot(3,1,1);
autocorr(zhat);
title("z_{jukf}");

subplot(3,1,2);
autocorr(abs(zhat));
title("|z_{jukf}|");

subplot(3,1,3);
autocorr(log(abs(zhat)));
title("log|z_{jukf}|");

figure;
subplot(3,1,1);
autocorr(zhat_nojukf);
title("z_{nojukf}");

subplot(3,1,2);
autocorr(abs(zhat_nojukf));
title("|z_{nojukf}|");

subplot(3,1,3);
autocorr(log(abs(zhat_nojukf)));
title("log|z_{nojukf}|");

figure;
subplot(3,1,1);
autocorr(u);
title("u")

subplot(3,1,2);
autocorr(abs(u));
title("|u|");

subplot(3,1,3);
autocorr(log(abs(u)));
title("log|u|");
