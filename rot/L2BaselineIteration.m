clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

maeList = zeros(Niter, 1);
rmseList = zeros(Niter, 1);
accList = zeros(Niter, 1);
for seed = 1:Niter
  disp(seed);
  params = mnistParameters();
  [rmse, mae, acc] = mnistL2Baseline(seed, params);
  maeList(seed) = mae;
  rmseList(seed) = rmse;
  accList(seed) = acc;
end


save('L2Stats');
