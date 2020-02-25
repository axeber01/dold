function [centroids, classLabels, valClassLabels] = randomDiscretizations(labels, c, N, M, valLabels)
% Discretize input labels in M different ways using N discrete bins, where
% the possible bin edges are contained in c

centroids = zeros(N, M);
classLabels = zeros(1, 1, N * M, length(labels));

for m = 1:M
  cm = randsample(c, N);
  cm = sort(cm);
  centroids(:, m) = cm;
  for i = 1:length(labels)
    d = abs(labels(i) - cm);
    [~, argmin] = min(d);
    classLabels(1, 1, (m - 1) * N + argmin, i) = 1;
  end
end

if nargin > 4
  valClassLabels = zeros(1, 1, N * M, length(valLabels));
  for m = 1:M
    cm = centroids(:, m);
    for i = 1:length(valLabels)
      d = abs(valLabels(i) - cm);
      [~, argmin] = min(d);
      valClassLabels(1, 1, (m - 1) * N + argmin, i) = 1;
    end
  end
else
  valClassLabels = [];
end

end

