function [rmse, mae, acc] = mnistL2Baseline(seed, params)

rng(seed);
addpath('../tools/');
[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;

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
  fullyConnectedLayer(1)
  regressionLayer];

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

net = trainNetwork(XTrain,YTrain,layers,options);

%%

YPredicted = predict(net,XValidation);

predictionError = double(YValidation) - double(YPredicted);

rmse = rms(predictionError);
mae = mean(abs(predictionError));
acc = mean(predictionError < 10);

end
