% for i = 1:5
%     for j = 1:5
%         r(i,j) = qstats_ukf(i,j).moments.kurtz;
%     end
% end

for i = 1:84
    v(i) = stats_jukf(i).moments.kurtz;
end
for i = 1:84
    n(i) = stats_nojukf(i).moments.kurtz;
end

mean(v)
std(v)
sum(v<0.05)
mean(n)
std(n)
sum(n<0.05)

figure;
subplot(1,2,1);
h = histogram(v,50);
title("jukf");
subplot(1,2,2);
g = histogram(n,50);
title("nojukf");