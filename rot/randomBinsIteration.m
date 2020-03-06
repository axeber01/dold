clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 10;

Nlist = 2.^[3, 4, 5, 6];
Mlist = 2.^[1, 2, 3, 4, 5, 6];
maeList = zeros(length(Nlist), length(Mlist), Niter);
rmseList = zeros(length(Nlist), length(Mlist), Niter);
accList = zeros(length(Nlist), length(Mlist), Niter);
for seed = 1:Niter
  for n = 1:length(Nlist)
    for m = 1:length(Mlist)
      disp([seed, n, m]);
      params = mnistParameters();
      params.N = Nlist(n);
      params.M = Mlist(m);
      [rmse, mae, acc] = mnistRandomBins(seed, params);
      maeList(n, m, seed) = mae;
      rmseList(n, m, seed) = rmse;
      accList(n, m, seed) = acc;
    end
  end
end

save('randomBinsStats');
