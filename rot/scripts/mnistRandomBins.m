function [rmse, mae, acc] = mnistRandomBins(seed, params)

rng(seed);
addpath('../tools/');
[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;

%% Create random classes
M = params.M;
N = params.N;
classes = (-45:45)';

[centroids, trainLabels, valLabels] = randomDiscretizations(YTrain, classes, N, M, YValidation);


%%

layers = [
  imageInputLayer([28 28 1], 'name', 'input')
  
  convolution2dLayer(3,8,'Padding','same', 'name', 'conv1')
  batchNormalizationLayer('name', 'bn1')
  reluLayer('name', 'relu1')
  
  averagePooling2dLayer(2,'Stride',2, 'name', 'ap1')
  
  convolution2dLayer(3,16,'Padding','same', 'name', 'conv2')
  batchNormalizationLayer('name', 'bn2')
  reluLayer('name', 'relu2');
  
  averagePooling2dLayer(2,'Stride',2, 'name', 'ap2')
  
  convolution2dLayer(3,32,'Padding','same', 'name', 'conv3')
  batchNormalizationLayer('name', 'bn3')
  reluLayer('name', 'relu3')
  
  convolution2dLayer(3,32,'Padding','same', 'name', 'conv4')
  batchNormalizationLayer('name', 'bn4')
  reluLayer('name', 'relu4')
  
  dropoutLayer(0.2, 'name', 'dropout')];

net = layerGraph(layers);
net = addLayers(net, concatenationLayer(3,M,'Name','concat'));

for m = 1:M
   namefc0 = ['fc_0_', num2str(m)];
  fc0 = fullyConnectedLayer(128, 'name', namefc0);
  namerelu = ['relu_0_', num2str(m)];
  relu = reluLayer('name', namerelu);
  
  namefc = ['fc', num2str(m)];
  fc = fullyConnectedLayer(N, 'name', namefc);
  namesm = ['sm', num2str(m)];
  sm = customSoftmaxLayer(namesm);
  
  net = addLayers(net, fc0);
  net = addLayers(net, relu);
  net = addLayers(net, fc);
  net = addLayers(net, sm);
  
  net = connectLayers(net, 'dropout', namefc0);
  net = connectLayers(net, namefc0, namerelu);
  net = connectLayers(net, namerelu, namefc);
  net = connectLayers(net, namefc, namesm);
  
  net = connectLayers(net, namesm, ['concat/in', num2str(m)]);
  
end

net = addLayers(net, smoothedCrossEntropyLayer('ce'));
net = connectLayers(net, 'concat', 'ce');

%%

options = trainingOptions(params.optimizer, ...
  'MiniBatchSize', params.miniBatchSize, ...
  'L2Regularization', params.L2reg, ...
  'MaxEpochs', params.maxEpochs, ...
  'InitialLearnRate', params.lr,...
  'LearnRateSchedule','piecewise',...
  'LearnRateDropFactor', params.lrDropFactor,...
  'LearnRateDropPeriod', params.lrDropPeriod,...
  'Shuffle','every-epoch',...
  'ValidationPatience',Inf,...
  'VerboseFrequency', 500, ...
  'Plots','none',...
  'Verbose',true);

net = trainNetwork(XTrain,trainLabels,net,options);

%% Predict validation data

ypred = predict(net,XValidation);

%% Equal weighting prediction

YPredicted = zeros(5000, 1);
for i =1:length(YPredicted)
  currY = ypred(i, :);
  currY = reshape(currY, [N, M]);
%   [~, argmax] = max(currY);
  YPredicted(i) = mean(sum(centroids .* currY));
%   for m = 1:M
%     YPredicted(i) = YPredicted(i) + 1 / M * centroids(argmax(m), m);
%   end
end


predictionError = double(YValidation) - double(YPredicted);

rmse = rms(predictionError);
mae = mean(abs(predictionError));
acc = mean(predictionError < 10);

end


