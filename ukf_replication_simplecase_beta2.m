clear; clc; close all;

%UKF_REPLICATION_SIMPLE_CASE:  Model is linear and Gaussian
% x(t)= b0+b1x(t-1)+b2q(t)   with b2 fixed, q iid N(0,1)
% y(t)= 0.5x(t)+v(t)         with v iid N(0,1)

rng(123);

sig = @(x) 1./(1+exp(-x));  %sigmoid function
invsig= @(x) -log(1./x-1);   %inverse sigmoid function

T = 20000;
stop = T;

x = zeros(1,T);
y = zeros(1,T);
q = randn(1,T);
v = log(abs(randn(1,T)));
%v= randn(1,T);

mu_v = -0.635;
sigma_v = sqrt(1.234);
S_v = -1.536;
%mu_v= 0;
%sigma_v= 1;
%S_v= 0;

b = [0.1; 0.8; 1];
Qnoise= [1e-6 1e-6 1e-6];
beta_hat = [0.2 0.9 0.8];
x(1)= b(1)/(1-b(2));          % initial value is s.s. mean
c = 0.5;                            % alpha in D matrix
y0= c*x(1);             % initial value for y
M = length(b);                  % excludes beta_2 for now
%Q = diag([b(end)^2 zeros(1,M)]);    % noise matrix
Q= diag([b(end)^2,Qnoise]);

a_hat = zeros(M+1,T);   % augmented state vectors over time
y_hat = zeros(1,T);     % measurements over time
inno = zeros(1,T);

a_hat(1,1) = 1;      % initial x approx value
a_hat(2:M+1,1) = [beta_hat(1); invsig(beta_hat(2)); log(beta_hat(3))];    % theta_hat
yhat(1)= c*a_hat(1,1);
inno(1) = 0;


% Initialization for loop
a_hatcorr = a_hat(:,1);
P_corr = [beta_hat(3)^2/(1-beta_hat(2)^2) 0 0 0; 0 0.5 0 0; 0 0 0.5 0; 0 0 0 0.5];
A = randn(4);
P_corr = 0.1*A*A';
P_corr = 0.1*ones(4) + 0.4*eye(4);

for t = 2:T
    % Actual values
    x(t) = [1 x(t-1) q(t)] * b;
    y(t) = c*x(t) + v(t);
    
    [a_hatpred, P_pred] = prediction(M, a_hatcorr, P_corr, Q);
    
    [a_hatcorr, P_corr, y_hat(t), g] = ...
        correction(M, c, [a_hatpred; mu_v], P_pred, sigma_v, S_v, y(t));
    inno(t) = g;
    a_hat(:,t) = a_hatcorr;
  
end

figure;
parcorr(a_hat(1,1:stop))
mdl = arima(1,0,0);
estimate(mdl,a_hat(1,1:stop)')

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

figure;
plot(t,inno);
xlim([0 T-1]);
title("inno");

figure;
plot(t,invsig(b(2))*ones(1,T),'b--',t,a_hat(3,:));
xlim([0 T-1]);
title("\theta_1");

a_hat(3,:) = sig(a_hat(3,:));
a_hat(4,:) = exp(a_hat(4,:));

figure;
for i = 1:M
    subplot(M,1,i);
    plot(t,b(i)*ones(1,T),'b--',t,a_hat(1+i,:));
    xlim([0 T-1]);
    title("\beta_" + i);
end


function [a_hatpred, P_pred] = prediction(M, mu, P, Q)
    sig = @(x) 1./(1+exp(-x));
%     mu(5) = 0;
%     a = sqrt(0.00001)*ones(1,4);
%     P1 = [chol(P) zeros(4,1);a sqrt(0.99996)];
%     P2 = P1*P1';
    P2 = [P zeros(4,1); zeros(1,4) 1];
    [w, chi] = create_gaussian_sigma_points([mu;0], P2);

    % Pass sigma points through process equation
    chi_x = chi(1,:);
    chi_theta = chi(2:M+1,:);
    chi_beta = [chi_theta(1,:); sig(chi_theta(2,:)); exp(chi_theta(3,:))];
    chi_q = chi(end,:);
%     q = randn;
    chi_pred = [sum(chi_beta .* [ones(1,length(chi_x)); chi_x; chi_q]); ...
        chi_theta];
    
    % Compute predicted augmented state and covariance matrix
    a_hatpred = chi_pred * w;
    P_pred = ...
        (chi_pred - a_hatpred) * diag(w) * (chi_pred - a_hatpred)';
        
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
    y_hat= chi_y*w;
    P_ay= (chi_a-mu(1:end-1))*diag(w)*(chi_y-y_hat)';
    P_yy= (chi_y-y_hat)*diag(w)*(chi_y-y_hat)';
    G= P_ay*inv(P_yy);
    a_hatcorr= mu(1:end-1)+G*(y-y_hat);
    P_corr= P-G*P_yy*G';
    g=1;
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