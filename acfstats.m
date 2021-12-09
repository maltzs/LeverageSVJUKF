% for i = 1:5
%     for j = 1:5
%         va(i,j) = qstats_ukf(i,j).moments.varz;
%     end
% end

for i = 1:100
    v(i) = stats_jukf(i).moments.varz;
end
for i = 1:100
    w(i) = stats_nojukf(i).moments.varz;
end

% m = cell(5,5);
% n = zeros(5,5);
t = cell(1,100);
u = zeros(1,100);
r = cell(1,100);
s = zeros(1,100);
% for i = 1:5
%     for j = 1:5
%         m{i,j} = qstats_ukf(i,j).acf.z.acfz;
%         n(i,j) = numel(find(abs(m{i,j}) > 2*sqrt(va(i,j)/2000)));
%     end
% end

for i = 1:100
    t{i} = stats_jukf(i).acf.z.acfz;
    u(i) = numel(find(abs(t{i}) > 2*sqrt(v(i)/10000)));
end
for i = 1:100
    r{i} = stats_nojukf(i).acf.z.acfz;
    s(i) = numel(find(abs(r{i}) > 2*sqrt(w(i)/10000)));
end

sum(u>1)
mean(u)
max(u)
min(u)

sum(s>1)
mean(s)
max(s)
min(s)