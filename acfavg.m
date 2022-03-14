sig = 2 / sqrt(T);
n = 100;

vz = [];
vabsz = [];
vlogabsz = [];
nz = [];
nabsz = [];
nlogabsz = [];
for i = 1:n
    for j = 1:21
        vz(i,j) = stats_jukf(i).acf.z.acfz(j);
        vabsz(i,j) = stats_jukf(i).acf.z.acfabsz(j);
        vlogabsz(i,j) = stats_jukf(i).acf.z.acflogabsz(j);
    end
end

a = any(isnan(vz)|isinf(vz)|isnan(vabsz)|isinf(vabsz)|isnan(vlogabsz)|isinf(vlogabsz),2);

for i = 1:n
    for j = 1:21
        nz(i,j) = stats_nojukf(i).acf.z.acfz(j);
        nabsz(i,j) = stats_nojukf(i).acf.z.acfabsz(j);
        nlogabsz(i,j) = stats_nojukf(i).acf.z.acflogabsz(j);
    end
end

b = any(isnan(nz)|isinf(nz)|isnan(nabsz)|isinf(nabsz)|isnan(nlogabsz)|isinf(nlogabsz),2);

c = a | b;
sum(c)

vz(c,:) = [];
vabsz(c,:) = [];
vlogabsz(c,:) = [];
nz(c,:) = [];
nabsz(c,:) = [];
nlogabsz(c,:) = [];

vz = mean(abs(vz));
vabsz = mean(abs(vabsz));
vlogabsz = mean(abs(vlogabsz));
nz = mean(abs(nz));
nabsz = mean(abs(nabsz));
nlogabsz = mean(abs(nlogabsz));

figure;
subplot(3,1,1);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,vz,'.','MarkerSize',15);
ylim([0 1]);
title("z_{jukf}");

subplot(3,1,2);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,vabsz,'.','MarkerSize',15);
ylim([0 1]);
title("|z_{jukf}|");

subplot(3,1,3);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,vlogabsz,'.','MarkerSize',15);
ylim([0 1]);
title("log|z_{jukf}|");

figure;
subplot(3,1,1);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,nz,'.','MarkerSize',15);
ylim([0 1]);
title("z_{nojukf}");

subplot(3,1,2);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,nabsz,'.','MarkerSize',15);
ylim([0 1]);
title("|z_{nojukf}|");

subplot(3,1,3);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,nlogabsz,'.','MarkerSize',15);
ylim([0 1]);
title("log|z_{nojukf}|");
