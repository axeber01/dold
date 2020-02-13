clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

fileID = fopen('ncLambda.txt','w');
for seed = 1:Niter
  params = utkParameters();
  [rmse, mae] = utkNc(seed, params);
  fprintf(fileID,'%s, %s\n',rmse, mae);
end

fclose(fileID);
