function params = mnistParameters()

%% Network training hyperparameters
params.optimizer = 'sgdm';
params.miniBatchSize = 128;
params.L2reg = 0.001;
params.lr = 0.001;  
params.lrDropFactor= 0.1; 
params.lrDropPeriod = 20;
params.maxEpochs = 30;

end
