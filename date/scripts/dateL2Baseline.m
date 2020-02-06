function [sAcc, sMae] = dateL2Baseline(seed, params)

%% Set random seed
rng(seed);

%% Create dataset
imds = imageDatastore(params.dataPath, ...
  'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsTest] = splitEachLabel(imds, params.split);

trainLabels = double(imdsTrain.Labels);
testLabels = double(imdsTest.Labels);

combinedTrain = table(imdsTrain.Files, trainLabels);
combinedTest = table(imdsTest.Files, testLabels);

aug = imageDataAugmenter('RandXReflection', params.randXReflection, ...
  'RandXTranslation', params.randXTranslation, 'RandYTranslation', params.randYTranslation, ...
  'RandRotation', params.randRotation, 'RandScale', params.randScale);


outputSize = [224, 224];
trainDs = augmentedImageDatastore(outputSize, combinedTrain, 'DataAugmentation', aug);
testDs = augmentedImageDatastore(outputSize, combinedTest);

%% Creat network

net = resnet50;
net = layerGraph(net);
net = removeLayers(net, 'fc1000');
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

net = addLayers(net, fullyConnectedLayer(2048, 'Name', 'fc1'));
net = addLayers(net, reluLayer('name', 'relu'));
net = addLayers(net, fullyConnectedLayer(1, 'Name', 'fc2'));
  
net = connectLayers(net, 'avg_pool', 'fc1');
net = connectLayers(net, 'fc1', 'relu');
net = connectLayers(net, 'relu', 'fc2');

net = addLayers(net, regressionLayer('Name', 'L2'));
net = connectLayers(net, 'fc2', 'L2');

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

YPredicted = predict(net, testDs);

%% Show results

sAcc = mean(round(YPredicted) == testLabels);
sMae = mean(abs(YPredicted - testLabels));
disp(['Test ACC: ', num2str(sAcc)]);
disp(['Test MAE: ', num2str(sMae)]);

end

