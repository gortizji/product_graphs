close all; clear all; clc

load dancer.mat

x0 = squeeze(mean(X,2));

X1 = X(:,:,1);
X2 = X(:,:,2);
X3 = X(:,:,3);

% Build spatial graph
param.use_full = 0;
param.k = 5;
G = gsp_nn_graph(x0,param);
G = gsp_compute_fourier_basis(G);

% Build product graph (time x space)
G = gsp_jtv_graph(G,size(X,2),[],param);

% Compute GFT basis
U2 = G.U;
U1 = dftmtx(T)/sqrt(T);

% Compute GFT
Xf1 = U2'*X1*conj(U1);
Xf2 = U2'*X2*conj(U1);
Xf3 = U2'*X3*conj(U1);

% Plot GFT
Xfg1 = max(20*log10(abs(Xf1)),[],1);
plot([-286:286],fftshift(smooth(Xfg1,10)))
hold on
Xfg2 = max(20*log10(abs(Xf3)),[],1);
plot([-286:286],fftshift(smooth(Xfg2,10)))
hold on
Xfg3 = max(20*log10(abs(Xf2)),[],1);
plot([-286:286],fftshift(smooth(Xfg3,10)))
axis([-0,286,20,120])
title('Maximum energy')
xlabel('Time frequency')
ylabel('dB')
figure
Xfg1 = max(20*log10(abs(Xf1)),[],2);
plot(smooth(Xfg1,100))
hold on
Xfg2 = max(20*log10(abs(Xf3)),[],2);
plot(smooth(Xfg2,100))
hold on
Xfg3 = max(20*log10(abs(Xf2)),[],2);
plot(smooth(Xfg3,100))
axis([0,1503,20,120])
title('Maximum energy')
xlabel('Spatial frequency')
ylabel('dB')

% Set bandwidth
K1 = 70;
K2 = 500;

% Crop GFT basis
U1_tilde = U1([1:K1/2, end-K1/2+1:end], :).';
U2_tilde = U2(:, 1:K2);

% Compute near-optimal sampler

select = greedy_kron_fp_min({U1_tilde, U2_tilde}, 600,[5, 5]);
L1 = select{1};
L2 = select{2};

X_est1 = real(reconstruct_sample_with_model(X1, U1_tilde, U2_tilde, L1, L2));
X_est2 = real(reconstruct_sample_with_model(X2, U1_tilde, U2_tilde, L1, L2));
X_est3 = real(reconstruct_sample_with_model(X3, U1_tilde, U2_tilde, L1, L2));

% Compute error
rmse = sqrt((norm(X1-X_est1,'fro')^2+ norm(X2-X_est2,'fro')^2+ norm(X3-X_est3,'fro')^2)/(norm(X1,'fro')^2+ norm(X2,'fro')^2+ norm(X3,'fro')^2));

% Show results
close all

subplot(222)
T = 100;
scatter3(X_est1(:,T), X_est3(:,T), X_est2(:,T), 'r.', 'SizeData', 30)
title('Estimated')
view([200, 20])
axis equal
axis off

subplot(221)
scatter3(X1(:,T), X3(:,T), X2(:,T), 'b.', 'SizeData', 30)
title('Original')
view([200, 20])
axis equal
axis off

subplot(224)
T = 300;
scatter3(X_est1(:,T), X_est3(:,T), X_est2(:,T),'r.', 'SizeData', 30)
title('Estimated')
view([200, 20])
axis equal
axis off

subplot(223)
scatter3(X1(:,T), X3(:,T), X2(:,T), 'b.', 'SizeData', 30)
title('Original')
view([200, 20])
axis equal
axis off 
