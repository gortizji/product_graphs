clear
close all
clc

% Load dataset
load movielens.mat

% Build users and movies graphs
G_u = gsp_graph(W_users);
G_m = gsp_graph(W_movies);

G_u = gsp_compute_fourier_basis(G_u);
G_m = gsp_compute_fourier_basis(G_m);

U2 = G_u.U;
U1 = G_m.U;

% Estimate ground truth with GRALS
[M,~,~] = grals(G_m.L,G_u.L,Otraining,M,10,Otest);

% Set GFT bandwidth
K1 = 20;
K2 = 20;

U1_tilde = U1(:,1:K1);
U2_tilde = U2(:,1:K2);

% Compute active query sample
select = greedy_kron_fp_min({U1_tilde,U2_tilde},45,[0,0]);

L1 = select{1};
L2 = select{2};

% Reconstruct ratings
M_hat = reconstruct_sample_with_model(M,U1_tilde, U2_tilde, L1, L2);

% Build unobserved nodes mask for evaluation
W = ones(G_u.N,G_m.N);
W(L2,L1) = 0;

% Compute RMSE estimate
rmse = sqrt(norm(W.*M-W.*M_hat,'fro')^2/sum(W(:)))

% Plot results
figure
subplot(141)
title('Observed data')
imagesc(M.*(1-W))
xlabel('Movies')
ylabel('Users')

subplot(142)
title('Ground truth')
imagesc(M)
xlabel('Movies')
ylabel('Users')

subplot(143)
title('Estimated data')
imagesc(M_hat)
xlabel('Movies')
ylabel('Users')

subplot(144)
title('Difference')
imagesc(M-M_hat)
xlabel('Movies')
ylabel('Users')