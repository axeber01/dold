function [sAcc, sMae] = dateMultipleBins(seed, params)

%% Set random seed
rng(seed);

%% Create dataset
imds = imageDatastore(params.dataPath, ...
  'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsTest] = splitEachLabel(imds, params.split);


%% Combine such that there is one unique class per head
Nclasses = 5;
trainLabels = double(imdsTrain.Labels);
testLabels = double(imdsTest.Labels);

Ntrain = length(trainLabels);
Ntest = length(testLabels);
combinedTrainLabels = [];
combinedTestLabels = [];
for n = 1:Nclasses
  if n == 1
    inds = trainLabels == 1;
    c = zeros(1, 1, 2, Ntrain);
    c(1, 1, 1, inds) = 1;
    c(1, 1, 2, ~inds) = 1;
  elseif n == 5
    inds = trainLabels == 5;
    c = zeros(1, 1, 2, Ntrain);
    c(1, 1, 1, ~inds) = 1;
    c(1, 1, 2, inds) = 1;
  else
    inds = trainLabels == n;
    indsLess = trainLabels < n;
    indsMore = trainLabels > n;
    c = zeros(1, 1, 3, Ntrain);
    c(1, 1, 1, indsLess) = 1;
    c(1, 1, 2, inds) = 1;
    c(1, 1, 3, indsMore) = 1;
  end
  combinedTrainLabels = cat(3, combinedTrainLabels, c);
  
  if n == 1
    inds = testLabels == 1;
    c = zeros(1, 1, 2, Ntest);
    c(1, 1, 1, inds) = 1;
    c(1, 1, 2, ~inds) = 1;
  elseif n == 5
    inds = testLabels == 5;
    c = zeros(1, 1, 2, Ntest);
    c(1, 1, 1, ~inds) = 1;
    c(1, 1, 2, inds) = 1;
  else
    inds = testLabels == n;
    indsLess = testLabels < n;
    indsMore = testLabels > n;
    c = zeros(1, 1, 3, Ntest);
    c(1, 1, 1, indsLess) = 1;
    c(1, 1, 2, inds) = 1;
    c(1, 1, 3, indsMore) = 1;
  end
  combinedTestLabels = cat(3, combinedTestLabels, c);

end

centroids = {{1, mean(2:5)}, {1, 2, mean(3:5)},{ mean(1:2), 3, mean(4:5)},{mean(1:3), 4, 5},{ mean(1:4), 5}};

%% Create new datastores

combinedTrain = table(imdsTrain.Files, squeeze(combinedTrainLabels)');
combinedTest = table(imdsTest.Files, squeeze(combinedTestLabels)');

aug = imageDataAugmenter('RandXReflection', params.randXReflection, ...
  'RandXTranslation', params.randXTranslation, 'RandYTranslation', params.randYTranslation, ...
  'RandRotation', params.randRotation, 'RandScale', params.randScale);


outputSize = [224, 224];
trainDs = augmentedImageDatastore(outputSize, combinedTrain, 'DataAugmentation', aug);
testDs = augmentedImageDatastore(outputSize, combinedTest);

%% Creat network

M = length(centroids);
net = resnet50;
net = layerGraph(net);
net = removeLayers(net, 'fc1000');
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

net = addLayers(net, fullyConnectedLayer(2048, 'name', 'fc2048'));
net = addLayers(net, reluLayer('name', 'relu'));
net = connectLayers(net, 'avg_pool', 'fc2048');
net = connectLayers(net, 'fc2048', 'relu');


net = addLayers(net, concatenationLayer(3,M,'Name','concat'));

for m = 1:M
  namefc = ['fc', num2str(m)];
  N = length(centroids{m});
  fc = fullyConnectedLayer(N, 'name', namefc);
  namesm = ['sm', num2str(m)];
  sm = customSoftmaxLayer(namesm);
  
  net = addLayers(net, fc);
  net = addLayers(net, sm);
  
  net = connectLayers(net, 'relu', namefc);
  net = connectLayers(net, namefc, namesm);
  
  net = connectLayers(net, namesm, ['concat/in', num2str(m)]);
  
end

net = addLayers(net, smoothedCrossEntropyLayer('ce'));
net = connectLayers(net, 'concat', 'ce');

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

%% Calculate conditional probability

probs = zeros(length(pred), 5);
probs(:, 1) = (pred(:, 1) + pred(:, 3) + pred(:, 6) / 2 + pred(:, 9) / 3 + pred(:, 12) / 4) / 5;
probs(:, 2) = (pred(:, 2) / 4 + pred(:, 4) + pred(:, 6) / 2  + pred(:, 9) / 3 + pred(:, 12) / 4) / 5;
probs(:, 3) = (pred(:, 2) / 4 + pred(:, 5) / 3 + pred(:, 7) + pred(:, 9) / 3 + pred(:, 12) / 4) / 5;
probs(:, 4) = (pred(:, 2) / 4 + pred(:, 5) / 3 + pred(:, 8) / 2 + pred(:, 10) + pred(:, 12) / 4) / 5;
probs(:, 5) = (pred(:, 2) / 4 + pred(:, 5) / 3 + pred(:, 8) / 2 + pred(:, 11) + pred(:, 13)) / 5;


[~, YPredicted] = max(probs');

%% Extract true labels
testLabels = double(imdsTest.Labels);

%% Show results

sAcc = mean(YPredicted' == testLabels);
sMae = mean(abs(YPredicted' - testLabels));
disp(['Test ACC: ', num2str(sAcc)]);
disp(['Test MAE: ', num2str(sMae)]);

end

