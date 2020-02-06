function params = dateParameters()

%% Data path
params.dataPath = '/usr/vision/axelb/HistoricalColor-ECCV2012/data/imgs/decade_database/';
params.split = 0.8;

%% Data augmentation
params.randXReflection = true;
params.randRotation = [-15, 15];
params.randXTranslation = [-20, 20];
params.randYTranslation = [-20, 20];
params.randScale = [0.7, 1.4];

%% Network training hyperparameters
params.optimizer = 'adam';
params.miniBatchSize = 32;
params.L2reg = 0.001;
params.lr = 0.0005;  
params.lrDropRate = 0.1; 
params.lrDropPeriod = 10;
params.validationFreq = 300;
params.maxEpochs = 30;

end
