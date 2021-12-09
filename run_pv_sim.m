function theta_min = run_pv_sim(T, N, trials, y, theta_0)
    sig = @(x) 1./(1+exp(-x));  %sigmoid function
    invsig= @(x) -log(1./x-1);   %inverse sigmoid function

    theta_0(2) = invsig(theta_0(2)); % transform phi

    L_min = Inf;
    for i = 1:trials
        fprintf("Trial " + i + "\n");
        [theta, L] = run_trial(T, N, y, theta_0);
        if L < L_min
            theta_min = theta;
            L_min = L;
        end
    end

    if isinf(L_min)
        theta_min = nan(1,numel(theta_0));
    end

    theta_min(2) = sig(theta_min(2));  % transform phi
end

function [theta, L] = run_trial(T, N, y, theta_0)
    global patience prev   %#ok

    patience = 0;
    prev = theta_0;
    
    pf_wrapper = @(theta) pf(y, T, N, theta);
    
    options = optimset('OutputFcn',@outfun);
    
    % Downhill simplex method from paper, no requirements on smoothness but
    % is unconstrained which does not work with some of the parameters
    [theta, L] = fminsearch(pf_wrapper,theta_0,options);
end

function L = pf(y, T, N, theta)
    [L, ~] = particle_filter(y, T, N, theta, false);
end

function stop = outfun(x, optimValues, varargin)
    global patience prev  %#ok

    stop = false;
    if norm(x-prev) < 1e-3
        patience = patience + 1;
        if patience >= 10 && optimValues.iteration > 75
            stop = true;
        end
    else
        patience = 0;
    end

    if isinf(optimValues.fval) || isnan(optimValues.fval)
        stop = true;
    end

    prev = x;
end