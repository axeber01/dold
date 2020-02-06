clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

fileID = fopen('multipleBins.txt','w');
for seed = 1:Niter
  params = dateParameters();
  [acc, mae] = dateMultipleBins(seed, params);
  fprintf(fileID,'%s, %s\n', acc, mae);
end

fclose(fileID);