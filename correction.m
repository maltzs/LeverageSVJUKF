function [a_hat_corr, P_corr] = correction(M, mu, P, sigma_nu, ...
    S_nu, y)
% CORRECTION The prediction step of both stochastic volatility joint
% unscented Kalman filter (SV-JUKF) models.
%   Inputs:
%   - M: Number of estimated parameters.
%   - mu: Current state prediction vector. Should be of length
%   M+2 where the last entry is mu_nu.
%   - P: Current prediction covariance. Must be (M+1)x(M+1) and
%   positive definite.
%   - sigma_nu: Standard deviation of nu.
%   - S_nu: Skewness of nu.
%   - y: The current measurement.
%
%   Outputs:
%   - a_hat_corr: Vector with the corrected state. Is of size M+1.
%   - P_corr: Corrected covariance. Is of size (M+1)x(M+1).
%
%   Throws:
%   - MATLAB:posdef: From create_sigma_points if P is not positive
%   definite.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    % Create sigma-points which take the distribution of nu into
    % account.
    [w, chi] = create_sigma_points(M, mu, P, sigma_nu, S_nu);
    
    % Augmented state sigma point components.
    chi_a = chi(1:end-1,:);

    chi_y = chi(M+2,:);    % measurement sigma point component
    y_hat = chi_y * w;

    P_ay = (chi_a - mu(1:end-1)) * diag(w) * (chi_y - y_hat)';
    P_yy = (chi_y - y_hat) * diag(w) * (chi_y - y_hat)';

    G = P_ay * P_yy^-1;    % Kalman gain

    a_hat_corr = mu(1:end-1) + G * (y-y_hat);
    P_corr = P - G * P_yy * G';
end

function [w, chi] = create_sigma_points(M, mu, C, sigma_nu, S_nu)
% CREATE_SIGMA_POINTS Creates sigma points that take the
% distribution of nu into account (Langner, 2022).
%   Inputs:
%   - M: Number of estimated parameters.
%   - mu: Current state estimate vector. Should be of length M+2
%   where the last entry is mu_nu.
%   - C: Current prediction covariance. Must be (M+1)x(M+1) and
%   positive definite.
%   - sigma_nu: Standard deviation of nu.
%   - S_nu: Skewness of nu.
%
%   Outputs:
%   - w: Sigma point weight vector. Length is 2(M+2).
%   - chi: Matrix of sigma points where each column is a
%   different sigma point. Size is (M+2)x(2(M+2)).
%
%   Throws:
%   - MATLAB:posdef: If C is not positive definite.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    a = 1/(M+2);

    % Gaussian components.
    s = sqrt(M+2)*ones(1,2*(M+2));
    w = (1/(2*(M+2)))*ones(2*(M+2),1);
    
    % Non-Gaussian components.
    s(M+2) = 0.5*(-S_nu+sqrt(S_nu^2+4/a));
    s(2*(M+2)) = 0.5*(S_nu+sqrt(S_nu^2+4/a));
    w(M+2) = a*s(2*(M+2))/(s(M+2)+s(2*(M+2)));
    w(2*(M+2)) = a*s(M+2)/(s(M+2)+s(2*(M+2)));
    
    chi_0 = [-diag(s(1:M+2)) diag(s(M+3:end))];
    D_inv = eye(M+2);
    D_inv(M+2,1) = 0.5;
    L = chol(C,'lower');
    
    A = zeros(M+2);
    A(1:M+1,1:M+1) = L;
    A(M+2,M+2) = sigma_nu;
    
    chi = D_inv * (A * chi_0 + mu);
end