classdef customSoftmaxLayer < nnet.layer.Layer
    methods
        function layer = customSoftmaxLayer(name)
            % Set layer name
            layer.Name = name;
            % Set layer description
            layer.Description = 'customSoftmaxLayer';
        end
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            X = X - max(X, [], 3);
            Z = exp(X) ./ sum(exp(X), 3);
        end
        function dLdX = backward(layer, X ,Z,dLdZ, ~)
            % Backward propagate the derivative of the loss function through
            % the layer
            N = size(Z, 4);
            M = size(Z, 3);
            dLdZ = squeeze(dLdZ);
            Z = squeeze(Z);
            for batch = 1:N
                temp = -Z(:, batch) * Z(:, batch)';
                temp(1:M+1:end) = temp(1:M+1:end) + Z(:, batch)';
                dLdX(1, 1, :, batch) = dLdZ(:, batch)' * temp;
            end
            
        end
    end
end