function z_hat = ukf_pure(T, u, y, theta)
    mu_v = -0.635;
    sigma_v = sqrt(1.234);
    S_v = -1.536;
        
    x_hat = zeros(1,T);   % augmented state vectors over time
    x_hat(1) = theta(1)/(1-theta(2));      % initial x approx value

    c = 0.5;                            % alpha in D matrix
    Q = 0.05;  % noise matrix
    P_corr = 0.5;
        
    % Initialization for loop
    x_hatcorr = x_hat(1);
    for t = 2:T
        epsilon = u(t-1)*exp(-x_hat(t-1)*c);
    
        [x_hatpred, P_pred, flag] = ...
            prediction(x_hatcorr, P_corr, Q, epsilon, theta);
        if flag > 0
            break
        end
        
        [x_hatcorr, P_corr, flag] = ...
            correction(c, [x_hatpred; mu_v], P_pred, sigma_v, S_v, y(t));
        if flag > 0
            break
        end
        
        x_hat(t) = x_hatcorr;
    end

    z_hat = exp(-x_hat/2) .* u;
end

function [a_hatpred, P_pred, flag] = prediction(mu, P, Q, epsilon, theta)
    [w, chi, flag] = create_gaussian_sigma_points(mu, P);

    if flag > 0
        a_hatpred = [];
        P_pred = [];
    else
        % Pass sigma points through process equation
        chi_pred = theta(1) + theta(2)*chi + ...
            theta(3)*((epsilon < 0) - 0.5) + theta(4)*epsilon + ...
            theta(5)*(abs(epsilon) - sqrt(2/pi));
        
        % Compute predicted augmented state and covariance matrix
        a_hatpred = chi_pred * w;
        P_pred = ...
            (chi_pred - a_hatpred) * diag(w) * (chi_pred - a_hatpred)'+Q;
    end        
end

function [a_hatcorr, P_corr, flag] = ...
    correction(c, mu, P, sigma_v, S_v, y)
    % Create sigma-points, including re-computing those for pred state
    [w, chi, flag] = create_sigma_points(c, mu, P, sigma_v, S_v);

    if flag > 0
        a_hatcorr = [];
        P_corr = [];
    else
        % Compute predicted vectors and correl mat
        % New w and sigma-points will match mu(state), P(state), no need to
        %    recompute
        chi_x= chi(1,:);
        chi_y= chi(end,:);
        y_hat= chi_y*w;
        P_xy= (chi_x-mu(1:end-1))*diag(w)*(chi_y-y_hat)';
        P_yy= (chi_y-y_hat)*diag(w)*(chi_y-y_hat)';
        G= P_xy*inv(P_yy);
        a_hatcorr= mu(1)+G*(y-y_hat);
        P_corr= P-G*P_yy*G';
    end
end

function [w, chi, flag] = create_gaussian_sigma_points(mu, C)
    [L, flag] = chol(C,'lower');
    if flag > 0
        w = [];
        chi = [];
    else
        chi = mu + [L -L];
        w = (1/2) * ones(2,1);
    end
end

function [w, chi, flag] = create_sigma_points(c, mu, C, sigma_v, S_v)
    a = 1/2;

    % Gaussian components
    s = sqrt(2) * ones(1,4);
    w = (1/4) * ones(4,1);
    
    % Non-gaussian components
    s(2) = 0.5*(-S_v + sqrt(S_v^2 + 4/a));
    s(4) = 0.5*(S_v + sqrt(S_v^2 + 4/a));
    w(2) = a*s(4)/(s(2)+s(4));
    w(4) = a*s(2)/(s(2)+s(4));
    
    chi_0 = [-diag(s(1:2)) diag(s(3:end))];
    Dinv = eye(2);
    Dinv(end,1) = c;
    [L, flag] = chol(C,'lower');
    if flag > 0
        w = [];
        chi = [];
    else
        A = zeros(2);
        A(1,1) = L;
        A(end,end) = sigma_v;
        
        chi = Dinv * (A * chi_0 + mu);
    end
end