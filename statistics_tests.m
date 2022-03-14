function stats = statistics_tests(u, z_hat)
    [stats.arch.archu, stats.arch.parchu] = archtest(u);
    [stats.arch.archz, stats.arch.parchz] = archtest(z_hat);
    
    [stats.lbq.lbq, stats.lbq.plbq] = lbqtest(z_hat.^2);
    
    stats.moments.meanz = mean(z_hat);
    stats.moments.varz = var(z_hat);
    stats.moments.skewz = skewness(z_hat);
    stats.moments.kurtz = kurtosis(z_hat);
    
    stats.acf.z.acfz = autocorr(z_hat);    
    stats.acf.z.acfabsz = autocorr(abs(z_hat));    
    stats.acf.z.acflogabsz = autocorr(z_hat.^2);
    
    stats.acf.u.acfu = autocorr(u);
    stats.acf.u.acfabsu = autocorr(abs(u));
    stats.acf.u.acflogabsu = autocorr(log(abs(u)));
end