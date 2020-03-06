function [rmse, mae, acc] = mnistCeBaseline(seed, params)

rng(seed);
addpath('../tools/');
[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;

Nclasses = 91;
classes = (-45:45)';
trainLabels = categorical(YTrain);
valLabels = categorical(YValidation);


%%

layers = [
  imageInputLayer([28 28 1])
  
  convolution2dLayer(3,8,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  averagePooling2dLayer(2,'Stride',2)
  
  convolution2dLayer(3,16,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  averagePooling2dLayer(2,'Stride',2)
  
  convolution2dLayer(3,32,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  convolution2dLayer(3,32,'Padding','same')
  batchNormalizationLayer
  reluLayer
  
  dropoutLayer(0.2)
  fullyConnectedLayer(Nclasses)
  softmaxLayer
  classificationLayer];

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

net = trainNetwork(XTrain,trainLabels,layers,options);

%%

y = predict(net,XValidation);
YPredicted = y * classes;

predictionError = double(YValidation) - double(YPredicted);

rmse = rms(predictionError);
mae = mean(abs(predictionError));
acc = mean(predictionError < 10);

end