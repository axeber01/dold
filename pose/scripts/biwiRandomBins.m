function [sRmse, sMae] = biwiRandomBins(seed, params)

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

%% Create discretization

N = 40;
M = 30;
classes = -78:0.01:78;
trainLabels = zeros(1, 1, N * M * 3, length(YTrain));
testLabels = zeros(1, 1, N * M * 3, length(YTest));
for n = 1:3
  [c, y, yt] = randomDiscretizations(YTrain(:, n), classes, N, M, YTest(:, n));
  %c = [-75 * ones(1, M); c; 75 * ones(1, M)];
  centroids{n} = c;
  trainLabels(1, 1, (n - 1) * N * M + 1:n * N * M, :) = y;
  testLabels(1, 1, (n - 1) * N * M + 1:n * N * M, :) = yt;
end

%% Create new datastores
outputSize = [224, 224];
aug = imageDataAugmenter('RandXTranslation', params.randXTranslation, ...
  'RandYTranslation', params.randYTranslation, 'RandScale', params.randScale);
trainDs = augmentedImageDatastore(outputSize, XTrain, trainLabels, 'DataAugmentation', aug);
testDs = augmentedImageDatastore(outputSize, XTest, testLabels);

%% Create Resnet50

Noutputs = 3;

net = resnet50;
net = layerGraph(net);
net = removeLayers(net, 'fc1000');
net = removeLayers(net, 'fc1000_softmax');
net = removeLayers(net, 'ClassificationLayer_fc1000');

% namefc = 'fc2048';
% namerelu = 'relu';
% net = addLayers(net, fullyConnectedLayer(2048, 'name', namefc));
% net = addLayers(net, reluLayer('name', namerelu));
% net = connectLayers(net, 'avg_pool', namefc);
% net = connectLayers(net, namefc, namerelu);

net = addLayers(net, concatenationLayer(3, 3 * M,'Name','concat'));

for n = 1:3
  namefc = ['fc', num2str(n)];
  namerelu = ['relu', num2str(n)];
  net = addLayers(net, fullyConnectedLayer(2048, 'name', namefc));
  net = addLayers(net, reluLayer('name', namerelu));
  net = connectLayers(net, 'avg_pool', namefc);
  net = connectLayers(net, namefc, namerelu);
  for m = 1:M
    namefc = ['fc', num2str(n), num2str(m)];
    fc = fullyConnectedLayer(N, 'name', namefc);
    namesm = ['sm', num2str(n), num2str(m)];
    sm = customSoftmaxLayer(namesm);
    
    net = addLayers(net, fc);
    net = addLayers(net, sm);
    
    net = connectLayers(net, namerelu, namefc);
    net = connectLayers(net, namefc, namesm);
    
    net = connectLayers(net, namesm, ['concat/in', num2str((n - 1) * M + m)]);
    
  end
end

net = addLayers(net, smoothedCrossEntropyLayer('ce'));
net = connectLayers(net, 'concat', 'ce');

%% Training options

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

%% Train network
[net, trainInfo] = trainNetwork(trainDs, net, options);

%% Evaluate
pred = predict(net, testDs);


%% Equal weighting prediction

YPredicted = zeros(length(pred), 3);
for i =1:length(YPredicted)
  for n = 1:3
    currY = pred(i, (n - 1) * N * M + 1:n * N * M);
    currY = reshape(currY, [N, M]);
%     c = centroids{n};
%     y = zeros(M, 1);
%     for m = 1:M
%       [~, maxInd] = max(currY(:, m));
%       y(m) = c(maxInd, m);
%     end
%     YPredicted(i, n) = mean(y);
    YPredicted(i, n) = mean(sum(centroids{n} .* currY));
  end
end

%% Show results

rmse = rms(YPredicted - YTest);
mae = mean(abs(YPredicted - YTest));
sRmse = ['Test RMSE: Yaw, ', num2str(rmse(1)), ', Pitch, ', num2str(rmse(2)), ...
  ', Roll, ', num2str(rmse(3)), ', Mean: ', num2str(mean(rmse))];
sMae = ['Test MAE: Yaw, ', num2str(mae(1)), ', Pitch, ', num2str(mae(2)), ...
  ', Roll, ', num2str(mae(3)), ', Mean: ', num2str(mean(mae))];
disp(sRmse);
disp(sMae);

save('randomBins');

end
