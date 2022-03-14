function leverage_SVJUKF1(T, jumps, M, theta, theta_hat, P_corr, Q, sp)
    sig = @(x) 1./(1+exp(-x));  %sigmoid function
    invsig= @(x) -log(1./x-1);   %inverse sigmoid function
    f = @(epsilon, alpha, gamma_1, gamma_2) alpha*((epsilon < 0) - 0.5) ...
        + gamma_1*epsilon + gamma_2*(abs(epsilon) - sqrt(2/pi));

    mu_v = -0.635;
    sigma_v = sqrt(1.234);
    S_v = -1.536;

    k = 1;

    if sp
        load("spall.mat","u");
        T = length(u);
        y = log(abs(u));
    else
        x = zeros(1,T+1);
        u = zeros(1,T);
        y = zeros(1,T);
        theta_plot = zeros(M,T);
            
        x(1) = theta(1,k)/(1-theta(2,k));          % initial value is s.s. mean
    end

    a_hat = zeros(M+1,T);   % augmented state vectors over time  
        
    a_hat(1,1) = theta_hat(1)/(1-theta_hat(2));      % initial x approx value
    a_hat(2:M+1,1) = [theta_hat(1); invsig(theta_hat(2)); theta_hat(3); theta_hat(4); theta_hat(5)];
    
    % Initialization for loop
    a_hatcorr = a_hat(:,1);
    if ~sp
        for t = 1:T
            if t == jumps(k)
                k = k+1;
            end
    
            % Actual values
            epsilon = randn;
            u(t) = exp(0.5*x(t))*epsilon;
            y(t) = log(abs(u(t)));
            x(t+1) = theta(1,k) + theta(2,k)*x(t) ...
                + f(epsilon, theta(3,k), theta(4,k), theta(5,k)) + randn*theta(6,k);
    
            theta_plot(:,t) = [theta(1,k) theta(2,k) theta(3,k) theta(4,k) theta(5,k)];
        end
    end
    
    for t = 2:T
        epsilon = u(t-1)*exp(-0.5*a_hat(1,t-1));
    
        [a_hatpred, P_pred] = prediction(M, a_hatcorr, P_corr, Q, epsilon, true);
        [a_hatcorr, P_corr] = ...
            correction(M, [a_hatpred; mu_v], P_pred, sigma_v, S_v, y(t));
    
        a_hat(:,t) = a_hatcorr;
    end
    
    a_hat(3,:) = sig(a_hat(3,:));
    
    x_hat = a_hat(1,:);
    z_hat = u.*exp(-0.5*x_hat);

    table(["Mean"; "Variance"; "Skewness"; "Kurtosis"],[mean(z_hat); var(z_hat); skewness(z_hat); kurtosis(z_hat)],'VariableNames',["Statistic","Value"])
    
    figure;
    qqplot(z_hat);
    title("");
    
    [~,plbq] = lbqtest(z_hat.^2);
    [~,parch] = archtest(z_hat);    
    table(["Lyung-Box Q"; "ARCH"],[plbq; parch],'VariableNames',["Test","p-value"])
    
    figure;
    subplot(2,1,1);
    autocorr(z_hat);
    title("$$\bf{\hat{z}}$$","Interpreter","latex");
    
    subplot(2,1,2);
    autocorr(z_hat.^2);
    title("$$\bf{\hat{z}^2}$$","Interpreter","latex");
    
    t = 0:T-1;
    standard_titles = ["\mu(1 - \phi)","\phi"];
    leverage_titles = ["\alpha","\gamma_1","\gamma_2"];
    figure;
    for i = 1:2
        subplot(2,1,i);
        hold on;
        if ~sp
            plot(t,theta_plot(i,:),'b--');
        end

        plot(t,a_hat(i+1,:));
        hold off;
        xlim([0 T-1]);
        title(standard_titles(i));
        xlabel("Time");
    end

    figure;
    for i = 1:3
        subplot(3,1,i);
        if ~sp
            plot(t,theta_plot(i+2,:),'b--');
        end
        
        plot(t,a_hat(i+3,:));
        hold off;
        xlim([0 T-1]);
        title(leverage_titles(i));
        xlabel("Time");
    end
end