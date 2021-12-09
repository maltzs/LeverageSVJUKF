%0 25 50 75 100
qind_pv = zeros(5);
qval_pv = zeros(5);
for i = 1:5
    [s,idx] = sort(theta_pv(:,i));
    qind_pv(1,i) = idx(1);
    qind_pv(2,i) = idx(24);
    qind_pv(3,i) = idx(48);
    qind_pv(4,i) = idx(72);
    qind_pv(5,i) = idx(96);

    qval_pv(1,i) = s(1);
    qval_pv(2,i) = s(24);
    qval_pv(3,i) = s(48);
    qval_pv(4,i) = s(72);
    qval_pv(5,i) = s(96);
end
qstats_pv = stats_pv(qind_pv);    