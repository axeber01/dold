function params = utkParameters()

%% Data path
% Full training set, no validation
params.dataPath = '/usr/vision/axelb/vpRegression/ageProj/utkDsCroppedFull.mat';

%% Data augmentation
% These are predefined in the trainDs
% params.randXTranslation = [-20, 20];
% params.randYTranslation = [-20, 20];
% params.randScale = [0.7, 1.4];
% params.randXFlip = true; 

%% Network training hyperparameters
params.optimizer = 'adam';
params.miniBatchSize = 32;
params.L2reg = 0.001;
params.lr = 0.0005;  
params.lrDropRate = 0.1; 
params.lrDropPeriod = 10;
params.validationFreq = 300;
params.maxEpochs = 30;
params.lambda = 0.01;
params.weights = [1, 2, 3, 4];

end