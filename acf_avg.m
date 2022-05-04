function acf_avg(T, stats)
% ACF_AVG Averages and plots the absolute value sample
% autocorrelations of the simulations.
%   Inputs:
%   - T: Time span of run.
%   - stats: A statistics structure from statistics_tests. Must
%   contain the following field:
%       - acf: A struct with sample autocorrelations. Must
%       contain the following fields:
%           - z: Sample autocorrelation of the residual series.
%           - z_squared: Sample autocorrelation of the squared
%           residual series.
%
%   Samuel Maltz, Master's thesis at The Cooper Union for the
%   Advancement of Science and Art (May 2022)

    sig = 2/sqrt(T);                    % significance bounds
    n = length(stats);
    n_lags = length(stats(1).acf.z);
    
    acf = zeros(n,n_lags,2);
    for i = 1:n
        for j = 1:n_lags
            acf(i,j,1) = stats(i).acf.z(j);
            acf(i,j,2) = stats(i).acf.z_squared(j);
        end
    end
    
    acf = mean(abs(acf),1);
    
    titles = ["$$\bf{\hat{z}}$$","$$\bf{\hat{z}^2}$$"];

    figure;
    for i = 1:2
        subplot(2,1,i);
        plot(0.5:0.5:n_lags-1,sig*ones(1,2*(n_lags-1)),'Color','b');
        hold on;
        stem(0:n_lags-1,acf(:,:,i),'.','MarkerSize',15,'Color','r');
        hold off;
        grid on;
        ylim([0 1]);
        xlabel("Lag");
        ylabel("Avg Abs Sample Autocorrelation");
        title(titles(i),'Interpreter','latex');
    end
end