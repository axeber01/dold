classdef smoothedCrossEntropyLayer < nnet.layer.RegressionLayer
    % Custom regression layer
    
    properties
      
    end
    
    methods
        function layer = smoothedCrossEntropyLayer(name)
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Calculates the cross entropy loss for smooth labels';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) 

            N = size(Y, 4);
            loss = -T .* log(max(Y, 1e-15)) + T .* log(max(T, 1e-15));
            loss = sum(loss(:)) / N;

            
        end
        
        function dLdY = backwardLoss(layer, Y, T)

            N = size(Y, 4);
            dLdY = -T ./ max(Y, 1e-15) / N;
        end
            
    end
end