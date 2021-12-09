clear; close all; clc;

rng(12345);

A = randn(3);
P_corr = [5 0 0; 0 0.5 0; 0 0 0.5];

x_0 = 0.01;
x_hat0 = x_0;

b = [0.1; 0.9; 0.25];
beta_hat = [0.1 0.9];


for i = 1:numel(x_0)

ukf_skewness_replication(x_0(i), x_hat0(i), b, beta_hat, P_corr);
end