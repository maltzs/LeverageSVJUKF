clear; clc; %close all;

%UKF_REPLICATION_SIMPLE_CASE:  Model is linear and Gaussian
% x(t)= b0+b1x(t-1)+b2q(t)   with b2 fixed, q iid N(0,1)
% y(t)= 0.5x(t)+v(t)         with v iid N(0,1)

rng(123);

sig = @(x) 1./(1+exp(-x));  %sigmoid function
invsig= @(x) -log(1./x-1);   %inverse sigmoid function

T = 20000;
start = 1;
stop = 20000;

x = zeros(1,T);
u = zeros(1,T);
y = zeros(1,T);
q = randn(1,T);
z= randn(1,T);

mu_v = -0.635;
sigma_v = sqrt(1.234);
S_v = -1.536;

b = [0.1; 0.8; 1];
Qnoise= [1e-6 1e-6];
beta_hat = [0.2 0.7];
x(1)= b(1)/(1-b(2));          % initial value is s.s. mean
c = 0.5;                            % alpha in D matrix
u(1) = exp(c * x(1)) * z(1);
y(1)= c*x(1);             % initial value for y
M = length(b) - 1;                  % excludes beta_2 for now
%Q = diag([b(end)^2 zeros(1,M)]);    % noise matrix
Q= diag([b(end)^2,Qnoise]);

a_hat = zeros(M+1,T);   % augmented state vectors over time
y_hat = zeros(1,T);     % measurements over time
inno = zeros(1,T);

a_hat(1,1) = 1;      % initial x approx value
a_hat(2:M+1,1) = [beta_hat(1); invsig(beta_hat(2))];    % theta_hat
yhat(1)= c*a_hat(1,1);
inno(1) = 0;


% Initialization for loop
P = [0.5 0 0; 0 0.5 0; 0 0 0.5];

for t = 2:T
    % Actual values
    x(t) = [1 x(t-1) q(t)] * b;
    u(t) = exp(c * x(t)) * z(t);
    y(t) = log(abs(u(t)));
    
    [a_hat(:,t), P, y_hat(t)] = ukf_step(M, a_hat(:,t-1), P, Q, c, mu_v, sigma_v, S_v, y(t));
end

x_hat = a_hat(1,:);

figure;
parcorr(x_hat(start:stop))
mdl = arima(1,0,0);
estimate(mdl,x_hat(start:stop)')

z_hat = u(start:stop) .* exp(-c * x_hat(start:stop));
% z_hat = log(z_hat);
[archu, parchu] = archtest(u);
fprintf("archtest of u: " + archu + " (pvalue: " + parchu + ")\n")
[archz, parchz] = archtest(z_hat);
fprintf("archtest of z_hat: " + archz + " (pvalue: " + parchz + ")\n")

[lbq, p] = lbqtest(z_hat);
fprintf("LB Q test: " + lbq + " (pvalue: " + p + ")\n")

figure;
qqplot(z_hat)

fprintf("Mean: " + mean(z_hat) + "\n")
fprintf("Variance: " + var(z_hat) + "\n")
fprintf("Skewness: " + skewness(z_hat) + "\n")
fprintf("Kurtosis: " + kurtosis(z_hat) + "\n")

figure;
subplot(3,1,1);
autocorr(z_hat);
title("z_{hat}");

subplot(3,1,2);
autocorr(abs(z_hat));
title("|z_{hat}|");

subplot(3,1,3);
autocorr(log(abs(z_hat)));
title("log|z_{hat}|");

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
plot(t,x,'b--',t,a_hat(1,:));
xlim([0 T-1]);
title("x");

subplot(2,1,2);
plot(t,y,'b--',t,y_hat);
xlim([0 T-1]);
title("y");

a_hat(3,:) = sig(a_hat(3,:));

figure;
for i = 1:M
    subplot(M,1,i);
    plot(t,b(i)*ones(1,T),'b--',t,a_hat(1+i,:));
    xlim([0 T-1]);
    title("\beta_" + i);
end


function [a_hatcorr, P_corr, y_hat] = ukf_step(M, mu, P, Q, c, mu_v, sigma_v, S_v, y)
    sig = @(x) 1./(1+exp(-x));

    % Create sigma-points, including re-computing those for pred state
    [w, chi] = create_sigma_points(M, [mu; mu_v], P, sigma_v, S_v);

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
       
    % Compute predicted vectors and correl mat
    % New w and sigma-points will match mu(state), P(state), no need to
    %    recompute
    chi_y= c*chi_pred(1,:) + chi(end,:);
    y_hat= chi_y*w;
    P_ay= (chi_pred-a_hatpred)*diag(w)*(chi_y-y_hat)';
    P_yy= (chi_y-y_hat)*diag(w)*(chi_y-y_hat)';
    G= P_ay*inv(P_yy);
    a_hatcorr= a_hatpred+G*(y-y_hat);
    P_corr= P_pred-G*P_yy*G';
end

function [w, chi] = create_sigma_points(M, mu, C, sigma_v, S_v)
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
    L = chol(C,'lower');
    
    A = zeros(M+2);
    A(1:M+1,1:M+1) = L;
    A(end,end) = sigma_v;
    
    chi = A * chi_0 + mu;
end