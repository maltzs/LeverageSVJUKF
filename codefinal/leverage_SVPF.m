function theta = leverage_SVPF(T, N_particles, M, y, theta_0)
% LEVERAGE_SVPF Runs leverage stochastic-volatility particle filter.
%   Inputs:
%   - T: Time span of run.
%   - N_particles: Number of particles to use in particle filter.
%   - M: Number of estimated parameters.
%   - y: Vector of log-return series. Must be length T.
%   - theta_0: Vector of initial theta estimates.
%
%   Outputs:
%   - theta: Vector of final theta estimates of length M.
%
%   Throws:
%   - MATLAB:nonaninf: If particle filter breaks down and returns NaN or
%   Inf.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the Advancement
%   of Science and Art (2022).

    % Global variables for optimization stopping condition.
    global patience prev    %#ok

    sig = @(x) 1./(1+exp(-x));     % sigmoid function
    invsig = @(x) -log(1./x-1);    % inverse sigmoid function

    sigma_eta_hat = theta_0(6);
    theta_0 = theta_0(1:M);
    theta_0(2) = invsig(theta_0(2));    % transforms phi to phi prime

    % Initializes optimization stopping condition variables.
    patience = 0;
    prev = theta_0;
    
    % First wrapper for particle filter function. Allows for additional
    % inputs.
    % Inputs:
    % - theta: theta parameters to optimize via particle filter.
    % Output:
    % Likelihood value from particle filter.
    pf_wrapper = @(theta) pf(y, T, N_particles, theta, sigma_eta_hat);

    % Options for optimizer to allow for early stopping.
    options = optimset('OutputFcn',@outfun);
    
    % Downhill simplex method, no requirements on smoothness but is
    % unconstrained. On off chance of model breakdown, redoes simulation
    % (see catch in leverage_SVJUKF_sim).
    [theta, L] = fminsearch(pf_wrapper,theta_0,options);
    if isnan(L) || isinf(L) || any(isnan(theta) | isinf(theta))
        throw(MException('MATLAB:nonaninf',"NaN or Inf"));
    end

    theta(2) = sig(theta(2));    % transform phi prime back to phi
end

function L = pf(y, T, N, theta, sigma_eta_hat)
% PF Second wrapper for particle filter. Allows for independent usage
% outside of optimizer.
%   Inputs:
%   - y: Vector of log-return series. Must be length T.
%   - T: Time span of run.
%   - N: Number of particles to use in particle filter.
%   - theta: Vector of theta values to use in particle filter iteration.
%   - sigma_eta_hat: Standard deviation of state process noise used in
%   particle filter.
%
%   Ouputs:
%   - L: Likelihood value from particle filter.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the Advancement
%   of Science and Art (2022).

    % Runs particle filter.
    [L, ~] = particle_filter(y, T, N, theta, sigma_eta_hat, true);
end

function stop = outfun(x, optimValues, varargin)
% OUTFUN Function run after every iteration of optimizer.
%   Inputs:
%   - x: Current value of optimized quantity (theta).
%   - optimValues: Struct with information about optimization process. Has
%   at least the following fields:
%       - iteration: Iteration count.
%       - fval: Current value of function being optimized.
%   - varargin: Additional inputs.
%
%   Outputs:
%   - stop: Logical value whether to stop optimization or not.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the Advancement
%   of Science and Art (2022).

    % Global variables for optimization stopping condition.
    global patience prev  %#ok

    % Stops optimization if past iteration 75 and 10 consecutive iterations
    % with change in x less than 1e-3.
    stop = false;
    if norm(x-prev) < 1e-3
        patience = patience + 1;
        if patience >= 10 && optimValues.iteration > 75
            stop = true;
        end
    else
        patience = 0;
    end

    % Stops optimization if on the off chance of model breakdown.
    if isinf(optimValues.fval) || isnan(optimValues.fval)
        stop = true;
    end

    prev = x;
end