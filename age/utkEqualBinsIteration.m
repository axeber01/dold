clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

fileID = fopen('equalBins.txt','w');
for seed = 1:Niter
  params = utkParameters();
  [rmse, mae] = utkEqualBins(seed, params);
  fprintf(fileID,'%s, %s\n',rmse, mae);
end

fclose(fileID);
