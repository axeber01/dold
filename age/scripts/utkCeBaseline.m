function [sRmse, sMae] = utkCeBaseline(seed, params)

%% Set random seed
rng(seed);

%% Load data
load(params.dataPath);

%% Create ResNet
Nclasses = 40;

net = resnet50;
net = layerGraph(net);

net = removeLayers(net, 'fc1000');
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

net = addLayers(net, fullyConnectedLayer(2048, 'Name', 'fc1'));
net = addLayers(net, reluLayer('name', 'relu'));
net = addLayers(net, fullyConnectedLayer(Nclasses, 'Name', 'fc2'));
  
net = connectLayers(net, 'avg_pool', 'fc1');
net = connectLayers(net, 'fc1', 'relu');
net = connectLayers(net, 'relu', 'fc2');

net = addLayers(net, softmaxLayer('Name', 'sm'));
net = connectLayers(net, 'fc2', 'sm');

net = addLayers(net, classificationLayer('Name', 'ce'));
net = connectLayers(net, 'sm', 'ce');

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
        'ValidationData', testDs, ...
        'ValidationFrequency', params.validationFreq, ...
        'ValidationPatience', Inf);

%% Train

[net, trainInfo] = trainNetwork(trainDs, net, options);

%% Prediction
pred = predict(net, testDs);

%% Take expected value
classes = (1:40)';
YPredicted = pred * classes;

%% Show results
rmse = rms(double(YPredicted)-double(testLabels));
mae = mean(abs(double(YPredicted)-double(testLabels)));
sRmse = ['Test RMSE, ', num2str(rmse)];
sMae = ['Test MAE, ', num2str(mae)];
disp(sRmse);
disp(sMae);

end
