G_u = gsp_graph(W_users);
G_m = gsp_graph(W_movies);

m = G_u.N;
n = G_m.N;
k = 10;
clc

Lw = G_u.L + 0.1 * eye(m);
Lh = G_m.L + 0.1 * eye(n);

Phi = Otraining;
%Phi = O;
[Y, W, H] = grals(Lh, Lw, Phi, M, k, Otest);

%%
G_u = gsp_compute_fourier_basis(G_u);
G_m = gsp_compute_fourier_basis(G_m);

U2 = G_u.U;
U1 = G_m.U;

K1 = 20;
K2 = 20;

U1_tilde = U1(:,1:K1);
U2_tilde = U2(:,1:K2);


%select = greedy_kron_logdet_max_lc({U1_tilde,U2_tilde},100,1e-6);
select = greedy_kron_fp_min({U1_tilde,U2_tilde},100,[5,5]);

L1 = select{1}; % Movies
L2 = select{2}; % Users

close all
M_hat = reconstruct_sample_with_model(Y,U1_tilde, U2_tilde, L1, L2);

W = ones(G_u.N,G_m.N);
W(L2,L1) = 0;

sqrt(norm(Otest.*Y - Otest.*M_hat, 'fro')^2/sum(Otest(:)))

%%
figure(1)
subplot(121)
imagesc(Y)
subplot(122)
imagesc(M_hat)
%%
sqrt(norm(Otest.*Y - Otest.*M_hat, 'fro')^2/sum(Otest(:)))

Crow = zeros(G_u.N,1);
Crow(L2) = 1;


Ccol = zeros(G_m.N,1);
Ccol(L1) = 1;
