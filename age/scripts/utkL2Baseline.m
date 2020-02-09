function [sRmse, sMae] = utkL2Baseline(seed, params)

%% Set random seed
rng(seed);

%% Load data
load(params.dataPath);

%% Create datastore for regression
sigma = 1; %std(double(trainLabels));
mu = mean(double(trainLabels));
ytrain = (double(trainLabels) - mu) / sigma;
ytest = (double(testLabels) - mu) / sigma;

trainTable = table(train{:,1}, ytrain);
testTable = table(test{:,1}, ytest);

outputSize = [224, 224];
aug = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', [-20,20], ...
  'RandYTranslation', [-20,20], 'RandScale', [0.7, 1.4]);
l2TrainDs = augmentedImageDatastore(outputSize, trainTable, 'ColorPreprocessing', 'gray2rgb', ...
  'DataAugmentation', aug);
l2TestDs = augmentedImageDatastore(outputSize, testTable, 'ColorPreprocessing', 'gray2rgb');

%% Create ResNet
Nclasses = 40;

net = resnet50;
net = layerGraph(net);
net = removeLayers(net, 'fc1000');
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

net = addLayers(net, fullyConnectedLayer(2048, 'Name', 'fc1'));
net = addLayers(net, reluLayer('name', 'relu'));
net = addLayers(net, fullyConnectedLayer(1, 'Name', 'fc2'));
net = addLayers(net, regressionLayer('Name', 'l2'));
  
net = connectLayers(net, 'avg_pool', 'fc1');
net = connectLayers(net, 'fc1', 'relu');
net = connectLayers(net, 'relu', 'fc2');
net = connectLayers(net, 'fc2', 'l2');

%% Training parameters


options = trainingOptions(params.optimizer, ...
        'MiniBatchSize', params.miniBatchSize, ...
        'MaxEpochs', params.maxEpochs, ...
        'InitialLearnRate', params.lr, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor', params.lrDropRate, ...
        'LearnRateDropPeriod', params.lrDropPeriod, ...
        'Shuffle','every-epoch', ...
        'Plots','none', ...
        'L2Regularization', params.L2reg, ...
        'VerboseFrequency', 300, ...
        'ValidationData', l2TestDs, ...
        'ValidationFrequency', params.validationFreq, ...
        'ValidationPatience', Inf);

%% Train

[net, trainInfo] = trainNetwork(l2TrainDs, net, options);

%% Prediction
ytHat = predict(net, l2TestDs);
YPredicted = ytHat * sigma + mu;


%% Show results
rmse = rms(double(YPredicted)-double(testLabels));
mae = mean(abs(double(YPredicted)-double(testLabels)));
sRmse = ['Test RMSE, ', num2str(rmse)];
sMae = ['Test MAE, ', num2str(mae)];
disp(sRmse);
disp(sMae);

end