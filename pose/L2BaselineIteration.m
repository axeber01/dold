clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

fileID = fopen('L2Baseline.txt','w');
for seed = 1:Niter
  params = biwiParameters();
  [rmse, mae] = biwiL2Baseline(seed, params);
  fprintf(fileID,'%s, %s\n', rmse, mae);
end

fclose(fileID);