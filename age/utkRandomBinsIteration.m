clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

fileID = fopen('randomBins.txt','w');
for seed = 1:Niter
  params = utkParameters();
  [rmse, mae] = utkRandomBins(seed, params);
  fprintf(fileID,'%s, %s\n',rmse, mae);
end

fclose(fileID);
