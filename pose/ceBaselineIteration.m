clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

fileID = fopen('ceBaseline.txt','w');
for seed = 1:Niter
  params = biwiParameters();
  [rmse, mae] = biwiCeBaseline(seed, params);
  fprintf(fileID,'%s, %s\n',rmse, mae);
end

fclose(fileID);