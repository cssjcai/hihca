classdef Polymd < dagnn.Filter
  properties
    method = 'sum'
    normalizeGradients = false;
  end

  methods
    function outputs = forward(obj, inputs, params)
            outputs{1} = polynomial_module_forward_pool(inputs{1}, params);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        [derInputs{1}, derParams] = polynomial_module_backward_pool(inputs{1}, derOutputs{1}, params);
        if obj.normalizeGradients   
            gradNorm = sum(abs(derInputs{1}(:))) + 1e-8;
            derInputs{1} = derInputs{1}/gradNorm;
        end
    end

    function obj = Polymd(varargin)
      obj.load(varargin) ;
    end
  end
end
