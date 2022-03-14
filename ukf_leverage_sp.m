clear; clc; close all;

%UKF_LEVERAGE:  Model is linear and Gaussian
% x(t)= mu*(1-phi) + phi*x(t-1) + f(epsilon, alpha, gamma_1, gamma_2) + eta
% with:
% f(epsilon, alpha, gamma_1, gamma_2) = alpha*(I(epsilon < 0) - 0.5)
%       + gamma_1*epsilon + gamma_2*(|epsilon| - sqrt(2/pi))
% and eta iid N(0,sigma2_eta)
% y(t)= 0.5x(t)+v(t)         with v iid log(|N(0,1)|)

sig = @(x) 1./(1+exp(-x));  %sigmoid function
invsig= @(x) -log(1./x-1);   %inverse sigmoid function
f = @(epsilon, alpha, gamma_1, gamma_2) alpha*((epsilon < 0) - 0.5) ...
        + gamma_1*epsilon + gamma_2*(abs(epsilon) - sqrt(2/pi));

load("spall.mat");
years = [0 19];
years = years + 1;
u = u(starts(years(1)):starts(years(2))-1);
T = length(u);
start = 1;
stop = T;

y = log(abs(u));

mu_v = -0.635;
sigma_v = sqrt(1.234);
S_v = -1.536;

Qnoise= [1e-6 1e-5 1e-7 1e-7 1e-8];

mu_hat = 0;
phi_hat = 0.9;
alpha_hat = 0.07;
gamma_1_hat = -0.08;
gamma_2_hat = 0.1;

theta_hat = [mu_hat*(1-phi_hat) phi_hat alpha_hat gamma_1_hat gamma_2_hat];
c = 0.5;                            % alpha in D matrix
M = length(theta_hat);

a_hat = zeros(M+1,T);   % augmented state vectors over time
y_hat = zeros(1,T);     % measurements over time
sigma_eta_hat = zeros(1,T);
sigma_eta = sqrt(0.16);
sigma_eta_hat(1) = sigma_eta - 0.1 + 0.2*rand;
sigma_eta_hat(1) = sigma_eta;
Q = diag([sigma_eta_hat(1)^2,Qnoise]);  % noise matrix
lambda = 0.9;

a_hat(1,1) = mu_hat;      % initial x approx value
% a_hat(2:M+1,1) = [theta_hat(1)*(1-theta_hat(2)); invsig(theta_hat(2)); theta_hat(3); theta_hat(4); theta_hat(5)];
a_hat(2:M+1,1) = [theta_hat(1); invsig(theta_hat(2)); theta_hat(3); theta_hat(4); theta_hat(5)];
y_hat(1) = c*a_hat(1,1);

% Initialization for loop
a_hatcorr = a_hat(:,1);
% P_corr = 0.5*eye(6);
P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);
varx = (alpha_hat^2/2 - sqrt(2/pi)*alpha_hat*gamma_1_hat + ...
    sqrt(2/pi)*alpha_hat*gamma_2_hat + gamma_1_hat^2 + ...
    gamma_2_hat^2 + sigma_eta_hat(1)^2)/(1 - phi_hat^2);

for t = 2:T
    epsilon = u(t-1)*exp(-a_hat(1,t-1)*c);

    [a_hatpred, P_pred] = prediction(M, a_hatcorr, P_corr, Q, epsilon);
    
    [a_hatcorr, P_corr, y_hat(t)] = ...
        correction(M, c, [a_hatpred; mu_v], P_pred, sigma_v, S_v, y(t));

    a = 4*(var(y(1:t)) - sigma_v^2);
    if a > 0
        varx = a;
    end
    newsigmaeta = sqrt(varx * (1-sig(a_hatcorr(3))^2) - a_hatcorr(4)^2/2 + sqrt(2/pi)*a_hatcorr(4)*a_hatcorr(5) - ...
    sqrt(2/pi)*a_hatcorr(4)*a_hatcorr(6) - a_hatcorr(5)^2 - ...
    a_hatcorr(6)^2);
    if isreal(newsigmaeta) && ~isnan(newsigmaeta) && ~isinf(newsigmaeta)
        sigma_eta_hat(t) = lambda*sigma_eta_hat(t-1) + (1-lambda)*newsigmaeta;
    else
        sigma_eta_hat(t) = sigma_eta_hat(t-1);
    end

%     m = a_hatcorr(2)/(1-sig(a_hatcorr(3)));
% %     m = a_hatcorr(2);
%     p = sig(a_hatcorr(3));
%     a = a_hatcorr(4);
%     g1 = a_hatcorr(5);
%     g2 = a_hatcorr(6);
%     i = 1:500;
%     newsigmaeta = sqrt((log(var(u(1:t))/prod(exp(-0.5*p.^(i-1)*a).*exp(-p.^(i-1)*g2*sqrt(2/pi)).*(exp(a*p.^(i-1)+p.^(2*i-2)*(g1-g2)^2/2).*normcdf(p.^(i-1)*(g2-g1))+exp(p.^(2*i-2)*(g1+g2)^2/2).*normcdf(p.^(2*i-2)*(g2+g1)))))-m)*(2*(1-p^2)));
%     if isreal(newsigmaeta) && ~isnan(newsigmaeta) && ~isinf(newsigmaeta)
%         sigma_eta_hat(t) = lambda*sigma_eta_hat(t-1) + (1-lambda)*newsigmaeta;
%     else
%         sigma_eta_hat(t) = sigma_eta_hat(t-1);
%     end        

%     Q(1,1) = sigma_eta_hat(t)^2;
    a_hat(:,t) = a_hatcorr;
end

x_hat = a_hat(1,:);

% figure;
% parcorr(x_hat(start:stop));
% mdl = arima(1,0,0);
% estimate(mdl,x_hat(start:stop)')

z_hat = u(start:stop) .* exp(-c * x_hat(start:stop));
[archy, parchy] = archtest(u);
fprintf("archtest of u: " + archy + " (pvalue: " + parchy + ")\n")
[archz, parchz] = archtest(z_hat);
fprintf("archtest of z_hat: " + archz + " (pvalue: " + parchz + ")\n")

[lbq, p] = lbqtest(z_hat.^2);
fprintf("LB Q test: " + lbq + " (pvalue: " + p + ")\n")

figure;
qqplot(z_hat);

fprintf("Mean: " + mean(z_hat) + "\n")
fprintf("Variance: " + var(z_hat) + "\n")
fprintf("Skewness: " + skewness(z_hat) + "\n")
fprintf("Kurtosis: " + kurtosis(z_hat) + "\n")

figure;
subplot(2,1,1);
autocorr(z_hat);
title("z_{hat}");

subplot(2,1,2);
autocorr(z_hat.^2);
title("z_{hat}^2");

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

t = 0:T-1;
figure;
subplot(2,1,1);
plot(t,x_hat);
xlim([0 T-1]);
title("x");

subplot(2,1,2);
plot(t,y,'b--',t,y_hat);
xlim([0 T-1]);
title("y");

a_hat(3,:) = sig(a_hat(3,:));
% a_hat(4,:) = exp(a_hat(4,:));
% a_hat(5,:) = -exp(a_hat(5,:));

t = 0:T-1;
titles = ["\mu" "\phi" "\alpha" "\gamma_1" "\gamma_2"];
figure;
subplot(2,1,1);
plot(t,a_hat(2,:));
xlim([0 T-1]);
title(titles(1));

subplot(2,1,2);
plot(t,a_hat(3,:));
xlim([0 T-1]);
title(titles(2));

figure;
subplot(2,1,1);
plot(t,a_hat(4,:));
xlim([0 T-1]);
title(titles(3));

subplot(2,1,2);
plot(t,a_hat(5,:));
xlim([0 T-1]);
title(titles(4));

figure;
subplot(2,1,1);
plot(t,a_hat(6,:));
xlim([0 T-1]);
title(titles(5));

subplot(2,1,2);
plot(t,sigma_eta_hat);
xlim([0 T-1]);
title("\sigma_{\eta}");


function [a_hatpred, P_pred] = prediction(M, mu, P, Q, epsilon)
    sig = @(x) 1./(1+exp(-x));

    [w, chi] = create_gaussian_sigma_points(mu, P);

    % Pass sigma points through process equation
    chi_x = chi(1,:);
    chi_theta_trans = chi(2:M+1,:);
    chi_theta = [chi_theta_trans(1,:); sig(chi_theta_trans(2,:)); ...
        chi_theta_trans(3,:); chi_theta_trans(4,:); ...
        chi_theta_trans(5,:)];
    chi_pred = [sum(chi_theta .* [ones(1,length(chi_x)); chi_x; ...
        ((epsilon < 0) - 0.5)*ones(1,length(chi_x)); ...
        epsilon*ones(1,length(chi_x)); ...
        (abs(epsilon) - sqrt(2/pi))*ones(1,length(chi_x))]); ...
        chi_theta_trans];
%     chi_pred = [sum(chi_theta .* [1 - chi_theta(2,:); chi_x; ...
%         ((epsilon < 0) - 0.5)*ones(1,length(chi_x)); ...
%         epsilon*ones(1,length(chi_x)); ...
%         (abs(epsilon) - sqrt(2/pi))*ones(1,length(chi_x))]); ...
%         chi_theta_trans];

    % Compute predicted augmented state and covariance matrix
    a_hatpred = chi_pred * w;
    P_pred = ...
        (chi_pred - a_hatpred) * diag(w) * (chi_pred - a_hatpred)'+Q;
        
end

function [a_hatcorr, P_corr, y_hat] = ...
    correction(M, c, mu, P, sigma_v, S_v, y)
% Create sigma-points, including re-computing those for pred state
    [w, chi] = create_sigma_points(M, c, mu, P, sigma_v, S_v);
    
    % Compute predicted vectors and correl mat
    % New w and sigma-points will match mu(state), P(state), no need to
    %    recompute
    chi_a= chi(1:end-1,:);
    chi_y= chi(end,:);
    y_hat= chi_y*w;
    P_ay= (chi_a-mu(1:end-1))*diag(w)*(chi_y-y_hat)';
    P_yy= (chi_y-y_hat)*diag(w)*(chi_y-y_hat)';
    G= P_ay*inv(P_yy);
    a_hatcorr= mu(1:end-1)+G*(y-y_hat);
    P_corr= P-G*P_yy*G';
end

function [w, chi] = create_gaussian_sigma_points(mu, C)
    M = length(mu);
    L = chol(C,'lower');
    chi = mu + sqrt(M) * [L -L];
    w = (1/(2*M)) * ones(2*M,1);
end

function [w, chi] = create_sigma_points(M, c, mu, C, sigma_v, S_v)
    a = 1/(M+2);

    % Gaussian components
    s = sqrt(M+2) * ones(1,2*(M+2));
    w = (1/(2*(M+2))) * ones(2*(M+2),1);
    
    % Non-gaussian components
    s(M+2) = 0.5*(-S_v + sqrt(S_v^2 + 4/a));
    s(2*(M+2)) = 0.5*(S_v + sqrt(S_v^2 + 4/a));
    w(M+2) = a*s(2*(M+2))/(s(M+2)+s(2*(M+2)));
    w(2*(M+2)) = a*s(M+2)/(s(M+2)+s(2*(M+2)));
    
    chi_0 = [-diag(s(1:M+2)) diag(s(M+3:end))];
    Dinv = eye(M+2);
    Dinv(end,1) = c;
    L = chol(C,'lower');
    
    A = zeros(M+2);
    A(1:M+1,1:M+1) = L;
    A(end,end) = sigma_v;
    
    chi = Dinv * (A * chi_0 + mu);
end