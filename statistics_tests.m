function stats = statistics_tests(z_hat, figs)
% STATISTICS_TESTS Runs various statistics on the residual series
% from the stochastic volatility joint unscented Kalman filter.
%   Inputs:
%   - z_hat: Vector of the residual series.
%   - figs: Logical value whether figures and tables should be
%   produced.
%
%   Outputs:
%   - stats: A struct with the statistics. Has the following fields:
%       - statistics: A struct with sample statistics. Has the
%       following fields:
%           - mean: Mean of z_hat.
%           - variance: Variance of z_hat.
%           - skewness: Skewness of z_hat.
%           - kurtosis: Kurtosis of z_hat.
%       - tests: A struct with results from tests for autoregressive
%       conditional heteroskedasticity (ARCH) effects in z_hat.
%       Has the following fields:
%           - p_lb: p-value from Ljung-Box test.
%           - p_arch: p-value from ARCH test.
%       - acf: A struct with sample autocorrelations. Has the
%       following fields:
%           - z: Sample autocorrelation of z_hat.
%           - z_squared: Sample autocorrelation of z_hat squared.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    % Sample statistics of z_hat.
    stats.statistics.mean = mean(z_hat);
    stats.statistics.variance = var(z_hat);
    stats.statistics.skewness = skewness(z_hat);
    stats.statistics.kurtosis = kurtosis(z_hat);

    % Tests for ARCH effects in z_hat.
    [~,stats.tests.p_lb] = lbqtest(z_hat.^2);
    [~,stats.tests.p_arch] = archtest(z_hat);

    % Sample autocorrelations of z_hat.
    stats.acf.z = autocorr(z_hat);    
    stats.acf.z_squared = autocorr(z_hat.^2);    

    if figs
        table(["Mean"; "Variance"; "Skewness"; "Kurtosis"], ...
            [stats.statistics.mean; stats.statistics.variance; ...
            stats.statistics.skewness; ...
            stats.statistics.kurtosis],'VariableNames', ...
            ["Statistic","Value"])
    
        % q-q plot of z_hat.
        figure;
        qqplot(z_hat);
        title("");
        
        table(["Ljung-Box"; "ARCH"],[stats.tests.p_lb; ...
            stats.tests.p_arch],'VariableNames',["Test","p-value"])

        figure;
        subplot(2,1,1);
        autocorr(z_hat);
        title("$$\bf{\hat{z}}$$",'Interpreter','latex');
        
        subplot(2,1,2);
        autocorr(z_hat.^2);
        title("$$\bf{\hat{z}^2}$$",'Interpreter','latex');
    end
end