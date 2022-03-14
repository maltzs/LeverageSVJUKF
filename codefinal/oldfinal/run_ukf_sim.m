function [theta_approx, zhat] = run_ukf_sim(T, R, u, y, theta_hat, start)
    sig = @(x) 1./(1+exp(-x));  %sigmoid function
    invsig= @(x) -log(1./x-1);   %inverse sigmoid function

    mu_v = -0.635;
    sigma_v = sqrt(1.234);
    S_v = -1.536;
    
    Qnoise= [1e-6 1e-5 1e-7 1e-7 1e-8];
    
    c = 0.5;                            % alpha in D matrix
    M = numel(theta_hat);
    
    a_hat = zeros(M+1,T*R);   % augmented state vectors over time
    sigma_eta_hat = zeros(1,T*R);
    sigma_eta_hat(1) = sqrt(0.05);%theta_hat(6);

    Q = diag([sigma_eta_hat(1)^2,Qnoise]);  % noise matrix
%     lambda = 0.9;
    
    P_corr = diag([0.5 0.01 0.1 0.01 0.01 0.01]);
%     varx = (theta_hat(3)^2/2 - sqrt(2/pi)*theta_hat(3)*theta_hat(4) + ...
%         sqrt(2/pi)*theta_hat(3)*theta_hat(5) + theta_hat(4)^2 + ...
%         theta_hat(5)^2 + sigma_eta_hat(1)^2)/(1 - theta_hat(2)^2);
   
    for i = 0:R-1
        a_hat(2:M+1,i*T+1) = [theta_hat(1); invsig(theta_hat(2)); ...
            theta_hat(3); theta_hat(4); theta_hat(5)];
        a_hat(1,i*T+1) = theta_hat(1)/(1-theta_hat(2));      % initial x approx value
        
        % Initialization for loop
        a_hatcorr = a_hat(:,i*T+1);

        for t = 2:T
            epsilon = u(t-1)*exp(-a_hat(1,i*T+t-1)*c);
        
            [a_hatpred, P_pred, flag] = ...
                prediction(M, a_hatcorr, P_corr, Q, epsilon);
            if flag > 0
                break
            end
            
            [a_hatcorr, P_corr, flag] = ...
                correction(M, c, [a_hatpred; mu_v], P_pred, sigma_v, S_v, ...
                y(t));
            if flag > 0
                break
            end
        
%             a = 4*(var(y(1:t)) - sigma_v^2);
%             if a > 0
%                 varx = a;
%             end
%             newsigmaeta = sqrt(varx * (1-sig(a_hatcorr(3))^2) - ...
%                 a_hatcorr(4)^2/2 + sqrt(2/pi)*a_hatcorr(4)*a_hatcorr(5) - ...
%             sqrt(2/pi)*a_hatcorr(4)*a_hatcorr(6) - a_hatcorr(5)^2 - ...
%             a_hatcorr(6)^2);
%             if isreal(newsigmaeta) && ~isnan(newsigmaeta) && ~isinf(newsigmaeta)
%                 sigma_eta_hat(t) = lambda*sigma_eta_hat(t-1) + (1-lambda)*newsigmaeta;
%             else
%                 sigma_eta_hat(t) = sigma_eta_hat(t-1);
%             end
%     
%             Q(1,1) = sigma_eta_hat(t)^2;
            a_hat(:,i*T+t) = a_hatcorr;
        end

        if flag > 0
            break
        end

        theta_hat = [a_hatcorr(2) sig(a_hatcorr(3)) a_hatcorr(4) ...
            a_hatcorr(5) a_hatcorr(6)];
    end

    if flag > 0
        theta_approx = nan(1,numel(theta_hat));
        zhat = nan(1,T-start+1);
    else
        a_hat(3,end) = sig(a_hat(3,end));
        theta_approx = a_hat(2:end,end)';
        x_hat = a_hat(1,:);
        zhat = u(start:end) .* exp(-c * x_hat(start:end));
    end
end

function [a_hatpred, P_pred, flag] = prediction(M, mu, P, Q, epsilon)
sig = @(x) 1./(1+exp(-x));  %sigmoid function

    [w, chi, flag] = create_gaussian_sigma_points(mu, P);

    if flag > 0
        a_hatpred = [];
        P_pred = [];
    else
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
    
        % Compute predicted augmented state and covariance matrix
        a_hatpred = chi_pred * w;
        P_pred = ...
            (chi_pred - a_hatpred) * diag(w) * (chi_pred - a_hatpred)'+Q;
    end        
end

function [a_hatcorr, P_corr, flag] = ...
    correction(M, c, mu, P, sigma_v, S_v, y)
    % Create sigma-points, including re-computing those for pred state
    [w, chi, flag] = create_sigma_points(M, c, mu, P, sigma_v, S_v);

    if flag > 0
        a_hatcorr = [];
        P_corr = [];
    else
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
end

function [w, chi, flag] = create_gaussian_sigma_points(mu, C)
    M = length(mu);
    [L, flag] = chol(C,'lower');
    if flag > 0
        w = [];
        chi = [];
    else
        chi = mu + sqrt(M) * [L -L];
        w = (1/(2*M)) * ones(2*M,1);
    end
end

function [w, chi, flag] = create_sigma_points(M, c, mu, C, sigma_v, S_v)
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
    [L, flag] = chol(C,'lower');
    if flag > 0
        w = [];
        chi = [];
    else
        A = zeros(M+2);
        A(1:M+1,1:M+1) = L;
        A(end,end) = sigma_v;
        
        chi = Dinv * (A * chi_0 + mu);
    end
end