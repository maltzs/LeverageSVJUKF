clear; close all; clc;

% GED distribution, nu = 2 is the normal case
% Unable to find way to sample this distribution, assuming normal for now
% lambda = @(nu) (2^(-2/nu)*gamma(1/nu)/gamma(3/nu))^(1/2);
% GEDpdf = @(epsilon, nu) (nu/(lambda(nu)*2^(1+1/nu)*gamma(1/nu))) ...
%     * exp(-abs(epsilon)^nu/(2*lambda(nu)^nu));

rs = ...
    RandStream('mt19937ar','Seed',123,'NormalTransform','Inversion');
RandStream.setGlobalStream(rs);

T = 500;
N = 1e4;
R = 100;
trials = 1;

% True parameters
mu = 0;
phi = 0.98;
alpha = 0.07;
gamma_1 = -0.08;
gamma_2 = 0.1;
sigma2_eta = 0.05;
% nu = 2;

theta_true = [mu*(1-phi) phi alpha gamma_1 gamma_2 sigma2_eta];

theta = zeros(R,numel(theta_true));
for i = 1:R
    fprintf("Simulation " + i + "\n");
    theta(i,:) = run_sim(T, N, trials, theta_true);
end

theta_avg = mean(theta,'omitnan');
theta_std = std(theta,'omitnan');


function theta_min = run_sim(T, N, trials, theta_true)
    sig = @(x) 1./(1+exp(-x));  %sigmoid function

    % Nonlinearity function
    f = @(epsilon, alpha, gamma_1, gamma_2) alpha*((epsilon < 0) - 0.5) ...
        + gamma_1*epsilon + gamma_2*(abs(epsilon) - sqrt(2/pi));

    mu = theta_true(1);
    phi = theta_true(2);
    alpha = theta_true(3);
    gamma_1 = theta_true(4);
    gamma_2 = theta_true(5);
    sigma2_eta = theta_true(6);

    % Approx parameters
    width = 0.6;
    mu_hat = mu - width/2 + width*rand;
    phi_hat = 1 - width + width*rand;
    alpha_hat = alpha - width/2 + width*rand;
    gamma_1_hat = gamma_1 - width/2 + width*rand;
    gamma_2_hat = gamma_2 - width/2 + width*rand;
    sigma2_eta_hat = sigma2_eta - width/2 + width*rand;

    % % Use to set all approx parameters manually
    % mu_hat = 0.1;
    % phi_hat = 0.9;
    % alpha_hat = 0.15;
    % gamma_1_hat = -0.15;
    % gamma_2_hat = 0.2;
    % sigma2_eta_hat = 0.1;
    
    theta_0 = [mu_hat*(1-phi_hat) phi_hat alpha_hat gamma_1_hat ...
        gamma_2_hat sigma2_eta_hat];
    
    h = zeros(1,T+1);
    y = zeros(1,T);
    
    h(1) = randn*sqrt(sigma2_eta^2/(1-phi^2)) + mu;
    for t = 1:T
        epsilon = randn;
        y(t) = exp(h(t)/2)*epsilon;
        h(t+1) = mu*(1-phi) + phi*h(t) ...
            + f(epsilon, alpha, gamma_1, gamma_2) + randn*sqrt(sigma2_eta);
    end
    
    L_min = Inf;
    for i = 1:trials
        fprintf("Trial " + i + "\n");
        [theta, L] = run_trial(T, N, f, y, theta_0, theta_true);
        if L < L_min
            theta_min = theta;
            L_min = L;
        end
    end

    if isinf(L_min)
        theta_min = nan(1,numel(theta_min));
    end

    theta_min(2) = sig(theta_min(2));

    close all;
end

function [theta, L] = run_trial(T, N, f, y, theta_0, theta_true)
    global ax an1 an2 patience   %#ok

    invsig= @(x) -log(1./x-1);   %inverse sigmoid function

    ax = cell(1,numel(theta_0));
    an1 = cell(1,numel(theta_0));
    an2 = cell(1,numel(theta_0));
    
    titles = ["\mu(1 - \phi)" "\phi" "\alpha" "\gamma_1" "\gamma_2" ...
        "\sigma^2_{\eta}"];
    for i = 0:1
        figure;
        for j = 1:numel(theta_0)/2
            ax{3*i+j} = subplot(3,1,j);
            ax{3*i+j}.Title.String = titles(3*i+j);
            ax{3*i+j}.XLabel.String = "Iteration";
            an1{3*i+j} = ...
                animatedline(ax{3*i+j},'Color','red','LineStyle','--');
            an1{3*i+j}.UserData = theta_true(3*i+j);  % stores true point
            an2{3*i+j} = animatedline(ax{3*i+j},'Color','blue');
            an2{3*i+j}.UserData = theta_0(3*i+j); % stores previous point
            hold(ax{3*i+j},'on');
        end
    end

    theta_0(2) = invsig(theta_0(2));

    patience = 0;
    
    pf_wrapper = @(theta) particle_filter(y, T, N, f, theta);
    
    % Constraints for some solvers
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = [-Inf -1 0 -Inf -Inf 0];
    ub = [Inf 1 Inf 0 Inf Inf];
    nonlcon = [];
    
    options = optimset('Display','iter','OutputFcn',@outfun);
    
    % Downhill simplex method from paper, no requirements on smoothness but
    % is unconstrained which does not work with some of the parameters
    [theta, L] = fminsearch(pf_wrapper,theta_0,options);
    
%     % Same as above but modified to accept constraints, MATLAB File
%     % Exchange function
%     [theta, L] = fminsearchbnd(pf_wrapper,theta_0,lb,ub,options);
    
%     % Gradient based solver, can accomidate constraints but needs function to
%     % be smooth, some values blow up after ~32 f-evals
%     [theta, L] = ...
%         fmincon(pf_wrapper,theta_0,A,b,Aeq,beq,lb,ub,nonlcon,options);
%     
%     % Gradient free, constrained solver
%     options1 = ...
%         optimset('Display','iter','OutputFcn',@outfun);
%     [theta, L] = patternsearch(pf_wrapper,theta_0,A,b,Aeq,beq,lb,ub, ...
%         nonlcon,options1);
    
    for i = 1:numel(theta_0)
        hold(ax{i},'off');
    end
end

function L = particle_filter(y, T, N, f, theta)
    sig = @(x) 1./(1+exp(-x));  %sigmoid function

    mutimes1mphi_hat = theta(1);
    phi_hat = sig(theta(2));
    alpha_hat = theta(3);
    gamma_1_hat = theta(4);
    gamma_2_hat = theta(5);
    sigma2_eta_hat = theta(6);

    L = 0;
    h_particles = randn(1,N)*sqrt(sigma2_eta_hat^2/(1-phi_hat^2)) + mutimes1mphi_hat/(1-phi_hat);
    for t = 2:T
        % Step 1
        h_particles_tilde = randn(1,N)*sqrt(sigma2_eta_hat) + ...
            mutimes1mphi_hat + phi_hat*h_particles + ...
            f(y(t-1)*exp(-h_particles/2), alpha_hat, gamma_1_hat, gamma_2_hat);
    
        % Step 2
        w = normpdf(y(t),0,exp(h_particles_tilde/2));
        L = L - log((1/N)*sum(w));
    
        % Step 3
        p = w / sum(w);

%         % Normal method of resampling
%         h_particles = randsample(h_particles_tilde,N,true,w);

        % Augmented steps of resampling to ensure continuous estimator
        % Step (i)
        [h_particles_hat, ind] = sort(h_particles_tilde);
        p_hat = p(ind);
    
        % Step (ii)
        u = zeros(1,N);
        v = rand(1,N);
        u(N) = v(N)^(1/N);
        for n = N-1:-1:1
            u(n) = u(n+1)*v(n)^(1/n);
        end
    
        % Step (iii)
        pi_hat = zeros(1,N+1);
        pi_hat(1) = p_hat(1)/2;
        pi_hat(N+1) = p_hat(N)/2;
        pi_hat(2:N) = (p_hat(2:N) + p_hat(1:N-1))/2;
    
        s = 0;
        j = 1;
        r = zeros(1,N);
        u_star = zeros(1,N);
        for i = 0:N
            s = s + pi_hat(i+1);
            while j <= N && u(j) <= s
                r(j) = i;
                u_star(j) = (u(j) - (s - pi_hat(i+1)))/pi_hat(i+1);
                j = j + 1;
            end
        end
    
        % Step (iv)
        for j = 1:N
            if r(j) == 0
                h_particles(j) = h_particles_hat(1);
            elseif r(j) == N
                h_particles(j) = h_particles_hat(N);
            else
                h_particles(j) = (h_particles_hat(r(j)+1) ...
                    - h_particles_hat(r(j)))*u_star(j) ...
                    + h_particles_hat(r(j));
            end
        end
    end
end

function stop = outfun(x, optimValues, varargin)
    global ax an1 an2 patience  %#ok

    sig = @(x) 1./(1+exp(-x));  %sigmoid function

    stop = false;
    prev = zeros(1,numel(x));
    y = x;
    y(2) = sig(y(2));
    for i = 0:1
        for j = 1:numel(x)/2
            hold(ax{3*i+j},'on');
            addpoints(an1{3*i+j},optimValues.iteration,an1{3*i+j}.UserData);
            addpoints(an2{3*i+j},optimValues.iteration,y(3*i+j));
            drawnow;

            prev(3*i+j) = an2{3*i+j}.UserData;
            an2{3*i+j}.UserData = y(3*i+j);
        end
    end

    if norm(y-prev) < 1e-3
        patience = patience + 1;
        if patience == 10  && optimValues.iteration > 100
            stop = true;
        end
    else
        patience = 0;
    end

    if isinf(optimValues.fval) || isnan(optimValues.fval)
        stop = true;
    end
end