function [L, z_hat] = particle_filter(y, T, N, theta, trans)
    sig = @(x) 1./(1+exp(-x));  %sigmoid function

    % Nonlinearity function
    f = @(epsilon, alpha, gamma_1, gamma_2) alpha*((epsilon < 0) - 0.5) ...
        + gamma_1*epsilon + gamma_2*(abs(epsilon) - sqrt(2/pi));
    
    mutimes1mphi_hat = theta(1);
    if trans
        phi_hat = theta(2);
    else
        phi_hat = sig(theta(2));
    end
        
    alpha_hat = theta(3);
    gamma_1_hat = theta(4);
    gamma_2_hat = theta(5);
    sigma_eta_hat = sqrt(0.05);

    L = 0;
    h_particles = randn(1,N)*sqrt(sigma_eta_hat^2/(1-phi_hat^2)) + ...
        mutimes1mphi_hat/(1-phi_hat);

    z_hat = zeros(1,T);
    z_hat(1) = mean(exp(h_particles/2))*y(1);
    for t = 2:T
        % Step 1
        h_particles_tilde = randn(1,N)*sigma_eta_hat + ...
            mutimes1mphi_hat + phi_hat*h_particles + ...
            f(y(t-1)*exp(-h_particles/2), alpha_hat, gamma_1_hat, ...
            gamma_2_hat);

        % Step 2
        w = normpdf(y(t),0,exp(h_particles_tilde/2));
        L = L - log((1/N)*sum(w));

        % Step 3
        p = w / sum(w);

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
        h_particles_hat1 = [h_particles_hat(1) h_particles_hat h_particles_hat(N)];
        h_particles = (h_particles_hat1(r+2) - h_particles_hat1(r+1)).*u_star + h_particles_hat1(r+1);

        z_hat(t) = mean(exp(h_particles/2))*y(t);
    end
end