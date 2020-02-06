function [sRmse, sMae] = biwiL2Baseline(seed, params)

%% Set random seed
rng(seed);

%% Load the data
trainData = load(params.trainPath);
testData = load(params.testPath);

XTrain = trainData.image;
XTrain = permute(XTrain, [2, 3, 4, 1]);

% Create copies with flipped images
XTrain = cat(4, XTrain, flip(XTrain, 2));

YTrain = trainData.pose;

% Flip yaw and roll for flipped images
yaw = YTrain(:, 1);
pitch = YTrain(:, 2);
roll = YTrain(:, 3);

YFlipped = [-yaw, pitch, -roll];
YTrain = [YTrain; YFlipped];

XTest = testData.image;
XTest = permute(XTest, [2, 3, 4, 1]);
YTest = testData.pose;

aug = imageDataAugmenter('RandXTranslation', params.randXTranslation, ...
  'RandYTranslation', params.randYTranslation, 'RandScale', params.randScale);
outputSize = [224, 224];
l2TrainDs = augmentedImageDatastore(outputSize, XTrain, YTrain, 'DataAugmentation', aug);
l2TestDs = augmentedImageDatastore(outputSize, XTest, YTest);


%% Create Resnet50

Noutputs = 3;
net = resnet50;
net = layerGraph(net);
net = removeLayers(net, 'fc1000');
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

net = addLayers(net, fullyConnectedLayer(2048, 'Name', 'fc1', ...
    'WeightLearnRateFactor', 1, 'BiasLearnRateFactor', 1));
net = addLayers(net, reluLayer('name', 'relu'));

net = addLayers(net, fullyConnectedLayer(Noutputs, 'Name', 'fc2', ...
    'WeightLearnRateFactor', 1, 'BiasLearnRateFactor', 1));
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

%% Train network
[net, trainInfo] = trainNetwork(l2TrainDs, net, options);

%% Evaluate
YPredicted = predict(net, l2TestDs);

%% Show results

rmse = rms(YPredicted - YTest);
mae = mean(abs(YPredicted - YTest));
sRmse = ['Test RMSE: Yaw, ', num2str(rmse(1)), ', Pitch, ', num2str(rmse(2)), ...
  ', Roll, ', num2str(rmse(3)), ', Mean: ', num2str(mean(rmse))];
sMae = ['Test MAE: Yaw, ', num2str(mae(1)), ', Pitch, ', num2str(mae(2)), ...
  ', Roll, ', num2str(mae(3)), ', Mean: ', num2str(mean(mae))];
disp(sRmse);
disp(sMae);

end
