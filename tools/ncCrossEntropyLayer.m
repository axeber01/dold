classdef ncCrossEntropyLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      lambda
      centroids
      weights
      
    end
    
    methods
        function layer = ncCrossEntropyLayer(name, lambda, centroids, weights)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Calculates the cross entropy loss and negative correlation';
            
            layer.lambda = lambda;
            layer.centroids = centroids;
            layer.weights = weights;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 

            Nce = size(Y, 4);
            ce = sum(-T .* log(max(Y, 1e-15)) + T .* log(max(T, 1e-15))) / Nce;
            
            N = size(layer.centroids, 1);
            M = size(layer.centroids, 2);
            
            nc = 0;
            for m = 1:M
              ym = Y(1, 1, (m - 1) * N + 1:m * N, :);
              for mm = m+1:M
                ymm = Y(1, 1, (mm - 1) * N + 1:mm * N, :);
                nc = nc - layer.lambda * layer.weights(mm - m) * ...
                  (sum(-ym .* log(max(ymm, 1e-15)) + ym .* log(max(ym, 1e-15))) + ...
                  sum(-ymm .* log(max(ym, 1e-15)) + ymm .* log(max(ymm, 1e-15)))) / (Nce * M);
              end
               
            end
            
            loss = sum(ce(:) + nc(:));

            
        end
        
%         function dLdY = backwardLoss(layer, Y, T)
% 
%             N = size(Y, 4);
%             dLdYce = -T ./ max(Y, 1e-15) / N;
%             
%             N = size(centroids, 1);
%             M = size(centroids, 2);
%             
%             dLdYnc = zeros(1, 1, N * M, Nce);
%             for m = 1:M
%               ym = Y(1, 1, (m - 1) * N + 1:m * N, :);
%               for mm = m+1:M
%                 ymm = Y(1, 1, (mm - 1) * N + 1:mm * N, :);
%                 dLdYnc = dLdYnc + layer.lambda * ...
%                   (sum(-ym .* log(max(ymm, 1e-15)) + ym .* log(max(ym, 1e-15))) + ...
%                   sum(-ymm .* log(max(ym, 1e-15)) + ymm .* log(max(ymm, 1e-15)))) / (Nce * M);
%               end
%                
%             end
%         end
            
    end
end