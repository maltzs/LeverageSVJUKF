clear; clc; close all;

%UKF_REPLICATION_SIMPLE_CASE:  Model is linear and Gaussian
% x(t)= b0+b1x(t-1)+b2q(t)   with b2 fixed, q iid N(0,1)
% y(t)= 0.5x(t)+v(t)         with v iid N(0,1)

sig = @(x) 1./(1+exp(-x));  %sigmoid function
invsig= @(x) -log(1./x-1);   %inverse sigmoid function

load("spall.mat");
years = [0 19];
years = years + 1;
u = u(starts(years(1)):starts(years(2)+1)-1);
T = length(u);
start = 1;
stop = T;

y = log(abs(u));

mu_v = -0.635;
sigma_v = sqrt(1.234);
S_v = -1.536;

Qnoise= [1e-6 1e-6];
beta_hat = [0.2 0.9 sqrt(0.36)];
c = 0.5;                            % alpha in D matrix
M = length(beta_hat) - 1;                  % excludes beta_2 for now
Q= diag([beta_hat(end)^2,Qnoise]);

a_hat = zeros(M+1,T);   % augmented state vectors over time
y_hat = zeros(1,T);     % measurements over time

a_hat(1,1) = beta_hat(1)/(1-beta_hat(2));      % initial x approx value
a_hat(2:M+1,1) = [beta_hat(1); invsig(beta_hat(2))];    % theta_hat
yhat(1)= c*a_hat(1,1);

% Initialization for loop
a_hatcorr = a_hat(:,1);
P_corr = [0.5 0 0; 0 0.5 0; 0 0 0.5];

for t = 2:T
    [a_hatpred, P_pred] = prediction(M, a_hatcorr, P_corr, Q);
    
    [a_hatcorr, P_corr, y_hat(t), g] = ...
        correction(M, c, [a_hatpred; mu_v], P_pred, sigma_v, S_v, y(t));
    a_hat(:,t) = a_hatcorr;
end

x_hat = a_hat(1,:);

z_hat = u(start:stop) .* exp(-c * x_hat(start:stop));
[archz, parchz] = archtest(z_hat);
fprintf("archtest of z_hat: " + archz + " (pvalue: " + parchz + ")\n")

[lbq, p] = lbqtest(z_hat.^2);
fprintf("LB Q test: " + lbq + " (pvalue: " + p + ")\n")

figure;
qqplot(z_hat);
title("");

fprintf("Mean: " + mean(z_hat) + "\n")
fprintf("Variance: " + var(z_hat) + "\n")
fprintf("Skewness: " + skewness(z_hat) + "\n")
fprintf("Kurtosis: " + kurtosis(z_hat) + "\n")

figure;
subplot(2,1,1);
autocorr(z_hat);
title("$$\bf{\hat{z}}$$","Interpreter","latex");

subplot(2,1,2);
autocorr(z_hat.^2);
title("$$\bf{\hat{z}^2}$$","Interpreter","latex");

t = 0:T-1;

a_hat(3,:) = sig(a_hat(3,:));

figure;
for i = 1:M
    subplot(M,1,i);
    plot(t,a_hat(1+i,:));
    xlim([0 T-1]);
    title("\beta_" + (i-1));
end


function [a_hatpred, P_pred] = prediction(M, mu, P, Q)
    sig = @(x) 1./(1+exp(-x));

    [w, chi] = create_gaussian_sigma_points(mu, P);

    % Pass sigma points through process equation
    chi_x = chi(1,:);
    chi_theta = chi(2:M+1,:);
    chi_beta = [chi_theta(1,:); sig(chi_theta(2,:))];
    chi_pred = [sum(chi_beta .* [ones(1,length(chi_x)); chi_x]); ...
        chi_theta];
    
    % Compute predicted augmented state and covariance matrix
    a_hatpred = chi_pred * w;
    P_pred = ...
        (chi_pred - a_hatpred) * diag(w) * (chi_pred - a_hatpred)'+Q;
        
end

function [a_hatcorr, P_corr, y_hat, g] = ...
    correction(M, c, mu, P, sigma_v, S_v, y)

    sig = @(x) 1./(1+exp(-x));
    invsig= @(x) -log(1./x-1);
    
% Create sigma-points, including re-computing those for pred state
    [w, chi] = create_sigma_points(M, c, mu, P, sigma_v, S_v);
    
    % Compute predicted vectors and correl mat
    % New w and sigma-points will match mu(state), P(state), no need to
    %    recompute
    chi_a= chi(1:end-1,:);
    chi_y= chi(end,:);
 %keyboard   
    y_hat= chi_y*w;
    P_ay= (chi_a-mu(1:end-1))*diag(w)*(chi_y-y_hat)';
    P_yy= (chi_y-y_hat)*diag(w)*(chi_y-y_hat)';
    G= P_ay*inv(P_yy);
    a = [1; 1; sig(mu(end-1))*(1-sig(mu(end-1)))];
    b = [1; 1; 1./(sig(mu(end-1))*(1-sig(mu(end-1))))];
    a_hatcorr= mu(1:end-1)+G*(y-y_hat);
    P_corr= P-G*P_yy*G';
    g = G(3)*(y-y_hat);
   %keyboard 
end

function [w, chi] = create_gaussian_sigma_points(mu, C)
    M = length(mu);
    %disp('g_sig_pts')
    L = chol(C,'lower');
    %C,L
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
    %chi_0
    Dinv = eye(M+2);
    Dinv(end,1) = c;
    L = chol(C,'lower');
    
    A = zeros(M+2);
    A(1:M+1,1:M+1) = L;
    A(end,end) = sigma_v;
    
    chi = Dinv * (A * chi_0 + mu);
   % disp('gen sig pts')
   % chi
   % w
    
end