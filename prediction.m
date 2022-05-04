function [a_hat_pred, P_pred] = prediction(M, mu, P, Q, z, ...
    leverage, theta)
% PREDICTION The prediction step of both stochastic volatility joint
% unscented Kalman filter (SV-JUKF) models.
%   Inputs:
%   - M: Number of estimated parameters.
%   - mu: Current state estimate vector.
%   - P: Current estimate covariance. Must be nxn where n is the
%   length of mu and positive definite.
%   - Q: Augmented state process noise matrix. Must have size
%   (M+1)x(M+1) and be symmetric.
%   - epsilon: Previous time step's residual. For use only when
%   leverage is true.
%   - leverage: Logical value whether the leverage SV-JUKF or
%   original SV-JUKF is in use. True represents the leverage SV-JUKF.
%   - theta: Vector with theta values for use when theta is not
%   estimated. For use only when M > 0.
%
%   Outputs:
%   - a_hat_pred: Vector with the predicted state. Is of size M+1.
%   - P_pred: Predicted covariance. Is of size (M+1)x(M+1).
%
%   Throws:
%   - MATLAB:posdef: From create_gaussian_sigma_points if P is
%   not positive definite.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    sig = @(x) 1./(1+exp(-x));    % sigmoid function

    % Creates standard sigma points.
    [w, chi] = create_gaussian_sigma_points(mu, P);

    % Passes sigma points through process equation.
    chi_x = chi(1,:);
    if M > 0
        chi_theta = chi(2:M+1,:);
        if leverage
            % Uses leverage SV-JUKF model.
            chi_phi_prime = chi_theta(2,:);

            % Transforms phi prime back to phi.
            chi_theta(2,:) = sig(chi_phi_prime);
            
            % First row is propagated x sigma point components,
            % other rows are same theta sigma point components.
            chi_pred = [sum(chi_theta.*[ones(1,length(chi_x)); ...
                chi_x; ((z < 0)-0.5)*ones(1,length(chi_x));
                z*ones(1,length(chi_x)); ...
                (abs(z)-sqrt(2/pi))*ones(1,length(chi_x))]); ...
                chi_theta];

            % Transforms phi to phi prime.
            chi_pred(3,:) = chi_phi_prime;
        else
            % Uses original SV-JUKF model. Transforms theta back
            % to beta.
            chi_beta = [chi_theta(1,:); sig(chi_theta(2,:))];
            
            % First row is propagated x sigma point components,
            % other rows are same theta sigma point components.
            chi_pred = [sum(chi_beta.*[ones(1,length(chi_x)); ...
                chi_x]); chi_theta];
        end
    else
        % Propagated x sigma points using the constant theta values.
        chi_pred = theta(1)+theta(2)*chi_x+theta(3)*((z < ...
            0)-0.5)+theta(4)*z+theta(5)*(abs(z)-sqrt(2/pi));
    end

    % Compute predicted augmented state and covariance matrix.
    a_hat_pred = chi_pred * w;
    P_pred = (chi_pred - a_hat_pred) * diag(w) * ...
        (chi_pred - a_hat_pred)' + Q;    
end

function [w, chi] = create_gaussian_sigma_points(mu, C)
% CREATE_GAUSSIAN_SIGMA_POINTS Creates standard Gaussian sigma
% points.
%   Inputs:
%   - mu: Current state estimate vector.
%   - C: Current estimate covariance. Must be nxn where n is
%   the length of mu and positive definite.
%
%   Outputs:
%   - w: Sigma point weight vector. Length is 2n where n is the
%   length of mu.
%   - chi: Matrix of sigma points where each column is a
%   different sigma point. Size is nx(2n) where n is the length
%   of mu.
%
%   Throws:
%   - MATLAB:posdef: If C is not positive definite.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    M = length(mu);
    L = chol(C,'lower');
    chi = mu + sqrt(M)*[L -L];
    w = (1/(2*M))*ones(2*M,1);
end