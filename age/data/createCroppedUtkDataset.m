rng(0);

% Path to UTKFace dataset
impath = '/usr/vision/axelb/utk/UTKFace/';

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = "_";

% Specify column names and types
opts.VariableNames = ["fileage", "VarName2", "VarName3", "VarName4"];
opts.VariableTypes = ["double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "VarName4", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "VarName4", "EmptyFieldRule", "auto");
opts = setvaropts(opts, "fileage", "TrimNonNumeric", true);
opts = setvaropts(opts, "fileage", "ThousandsSeparator", ",");

% Import the data from the csv tables
utktrain = readtable("utk_train.csv", opts);
utktest = readtable("utk_test.csv", opts);


%% Find train/test split
testNames = cell(size(utktest, 1), 1);
testLabels = zeros(size(utktest, 1), 1);

for i = 1:size(utktest, 1)
  gender = mat2str(utktest(i, 2).Variables);
  race = mat2str(utktest(i, 3).Variables);
  temp = str2mat(utktest(i, 4).Variables);
  age = temp(32:end);
  name = temp(1:30);
  
  testNames{i} = [impath, mat2str(str2double(age)+21), '_', gender, '_', race, '_', name];
  testLabels(i) = str2double(age);
end

%%
trainNames = cell(size(utktrain, 1), 1);
trainLabels = zeros(size(utktrain, 1), 1);

missingindex = [];
for i = 1:size(utktrain, 1)
  gender = mat2str(utktrain(i, 2).Variables);
  race = mat2str(utktrain(i, 3).Variables);
  temp = str2mat(utktrain(i, 4).Variables);
  temp(isspace(temp)) = '.';
  if ~isempty(temp)
    age = temp(32:end);
    name = temp(1:30);
    
    trainNames{i} = [impath, mat2str(str2double(age)+21), '_', gender, '_', race, '_', name];
    trainLabels(i) = str2double(age);
    if strcmp(trainNames{i}(end), 'p')
      trainNames{i} = [trainNames{i}, 'g'];
    elseif strcmp(trainNames{i}(end), ',')
      trainNames{i} = trainNames{i}(1:end-1);
    end
    
  else
    missingindex = [missingindex, i];
  end
  
end

%% Remove corrupted files
trainNames(missingindex) = [];
trainLabels(missingindex) = [];
trainNames(695) = [];
trainLabels(696) = [];
trainNames(11051) =  [];
trainLabels(11051) = [];

%% Extract Labels and create new data stores

imdsTrain = imageDatastore(trainNames);
imdsTest = imageDatastore(testNames);

trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

train = table(imdsTrain.Files, trainLabels);
test = table(imdsTest.Files, testLabels);

%% Create augmented image datastores
outputSize = [224, 224];

aug = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', [-20,20], ...
  'RandYTranslation', [-20,20], 'RandRotation', [0, 0], 'RandScale', [0.7, 1.4]);
trainDs = augmentedImageDatastore(outputSize, train, 'ColorPreprocessing', 'gray2rgb', ...
  'DataAugmentation', aug);
testDs = augmentedImageDatastore(outputSize, test, 'ColorPreprocessing', 'gray2rgb');


%% Save

save('utkDsCroppedFull', 'trainDs', 'testDs', ...
  'train', 'test', 'trainLabels', 'testLabels');
