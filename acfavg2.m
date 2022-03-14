sig = 2 / sqrt(2000);

vz = [];
vabsz = [];
vlogabsz = [];
nz = [];
nabsz = [];
nlogabsz = [];
for i = 1:100
    for j = 1:21
        vz(i,j) = stats_ukf(i).acf.z.acfz(j);
        vabsz(i,j) = stats_ukf(i).acf.z.acfabsz(j);
        vlogabsz(i,j) = stats_ukf(i).acf.z.acflogabsz(j);
    end
end

a = any(isnan(vz)|isinf(vz)|isnan(vabsz)|isinf(vabsz)|isnan(vlogabsz)|isinf(vlogabsz),2);

for i = 1:96
    for j = 1:21
        nz(i,j) = stats_pv(i).acf.z.acfz(j);
        nabsz(i,j) = stats_pv(i).acf.z.acfabsz(j);
        nlogabsz(i,j) = stats_pv(i).acf.z.acflogabsz(j);
    end
end

b = any(isnan(nz)|isinf(nz)|isnan(nabsz)|isinf(nabsz)|isnan(nlogabsz)|isinf(nlogabsz),2);

vz(a,:) = [];
vabsz(a,:) = [];
vlogabsz(a,:) = [];
nz(b,:) = [];
nabsz(b,:) = [];
nlogabsz(b,:) = [];

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
title("z_{ukf}");

subplot(3,1,2);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,vabsz,'.','MarkerSize',15);
ylim([0 1]);
title("|z_{ukf}|");

subplot(3,1,3);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,vlogabsz,'.','MarkerSize',15);
ylim([0 1]);
title("log|z_{ukf}|");

figure;
subplot(3,1,1);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,nz,'.','MarkerSize',15);
ylim([0 1]);
title("z_{pf}");

subplot(3,1,2);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,nabsz,'.','MarkerSize',15);
ylim([0 1]);
title("|z_{pf}|");

subplot(3,1,3);
plot(1:20,sig*ones(1,20));
hold on;
stem(0:20,nlogabsz,'.','MarkerSize',15);
ylim([0 1]);
title("log|z_{pf}|");
