function [sAcc, sMae] = dateCeBaseline(seed, params)

%% Set random seed
rng(seed);

%% Create dataset
imds = imageDatastore(params.dataPath, ...
  'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsTest] = splitEachLabel(imds, params.split);

aug = imageDataAugmenter('RandXReflection', params.randXReflection, ...
  'RandXTranslation', params.randXTranslation, 'RandYTranslation', params.randYTranslation, ...
  'RandRotation', params.randRotation, 'RandScale', params.randScale);


outputSize = [224, 224];
trainDs = augmentedImageDatastore(outputSize, imdsTrain, 'DataAugmentation', aug);
testDs = augmentedImageDatastore(outputSize, imdsTest);

%% Creat network

Nclasses = 5;

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

YPredicted = classify(net, testDs);

%% Extract true labels
testLabels = double(imdsTest.Labels);

%% Show results

sAcc = mean(double(YPredicted) == testLabels);
sMae = mean(abs(double(YPredicted) - testLabels));
disp(['Test ACC: ', num2str(sAcc)]);
disp(['Test MAE: ', num2str(sMae)]);

end

