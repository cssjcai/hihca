function net = hihca_model_hed(net, imdb, opts)

%% hed-based integration
% ------------------------------------------------------------------------------------
assert(numel(opts.hieLayerName)==numel(opts.rescaleLayerFactor),...
    'You must assign scaling factors for all layers !');

hca_inputs = cell(1, numel(opts.hieLayerName));
hca_layer_names = cell(1, numel(opts.hieLayerName));
hca_fdims = cell(1, numel(opts.hieLayerName));

for l = 1:numel(opts.hieLayerName)
    
    % layer scaling
    ca_input = net.layers(opts.hieLayerIDX(l)).outputs;
    sca_output = ['sc_',num2str(l)];
    layer_name = ['scale_',num2str(l)];
    param_name = ['scalar_',num2str(l)];
    param_value = single(opts.rescaleLayerFactor(l));
    
    net.addLayer(layer_name, dagnn.Scale('hasBias', false), ca_input, sca_output, {param_name});
    
    var_idx = net.getParamIndex(param_name);
    net.params(var_idx).value = param_value;
    net.params(var_idx).learningRate = 10;
    
    % degree-1
    input = sca_output;
    output = ['l', num2str(l), '_', 'pm1'];
    layer_name = ['layer_', num2str(l), '_', 'polymod_1'];
    net.addLayer(layer_name, dagnn.Pooling('poolSize', ...
        opts.imageScale*[opts.whc(1,l), opts.whc(2,l)], 'method', 'avg'), input, output);
    
    % degree-r
    hi_input = {output};
    hi_layer_names = {layer_name};
    hi_fdims = opts.whc(3,l);
    
    if opts.kernelDegree > 1
        
        pm_dims = opts.num1x1Filter .* ones(1, opts.kernelDegree-1);
        
        for d = 2 : opts.kernelDegree
            
            input = sca_output;
            output = ['l', num2str(l), '_', 'pm',num2str(d)];
            layer_name = ['layer_', num2str(l), '_', 'polymod_', num2str(d)];
            
            switch opts.init1x1Filter
                case 'rad'
                    if(~exist(opts.init1x1FilterDir, 'dir'))
                        mkdir(opts.init1x1FilterDir);
                    end
                    
                    if exist(fullfile(opts.init1x1FilterDir, ['l', num2str(l), '_', 'd', num2str(d), '_', 'radW.mat']), 'file') == 2
                        fprintf('loading Rademacher weight from saved file.\n');
                        load(fullfile(opts.init1x1FilterDir, ['l', num2str(l), '_', 'd', num2str(d), '_', 'radW.mat']));
                    else
                        factor=1.0/sqrt(pm_dims(d-1));
                        fprintf('generating new Rademacher weight\n');
                        
                        init_1x1Filters = cell(1,d);
                        for s = 1:d
                            init_1x1Filters{s} = single(factor*(randi(2,opts.whc(3,l), pm_dims(d-1))*2-3));
                        end
                        savefast(fullfile(opts.init1x1FilterDir, ['l', num2str(l), '_', 'd', num2str(d), '_', 'radW.mat']), 'init_1x1Filters');
                    end
                    
                case 'random'
                    if(~exist(opts.init1x1FilterDir, 'dir'))
                        mkdir(opts.init1x1FilterDir);
                    end
                    
                    if exist(fullfile(opts.init1x1FilterDir, ['d', num2str(d), '_', 'ramW.mat']), 'file') == 2
                        fprintf('loading Random weight from saved file.\n');
                        load(fullfile(opts.init1x1FilterDir, ['d', num2str(d), '_', 'ramW.mat']));
                    else
                        factor = 1.0/sqrt(pm_dims(d-1)); %0.001
                        fprintf('generating new Random weight\n');
                        
                        init_1x1Filters = cell(1,d);
                        for s = 1:d
                            init_1x1Filters{s} = single(factor *randn(opts.whc(3,l), pm_dims(d-1)));
                        end
                        savefast(fullfile(opts.init1x1FilterDir, ['l', num2str(l), '_', 'd', num2str(d), '_', 'radW.mat']), 'init_1x1Filters');
                    end
                    
            end
            
            hi_param_names = {};
            
            for s = 1:d
                param_name = ['l', num2str(l), 'conv_d',num2str(d),'fm',num2str(s)];
                hi_param_names = cat(2, hi_param_names, param_name);
            end
            
            net.addLayer(layer_name, Polymd(), input, output, hi_param_names);
            
            for s = 1:d,
                var_idx = net.getParamIndex(hi_param_names{s});
                param_value = init_1x1Filters{s};
                net.params(var_idx).value = param_value;
                switch opts.init1x1Filter
                    case 'rad'
                        net.params(var_idx).learningRate = 1;
                    case 'ramdom'
                        net.params(var_idx).learningRate = 10;
                end
            end
            
            hi_input = cat(2, hi_input, {output});
            hi_layer_names = cat(2, hi_layer_names, {layer_name});
            hi_fdims = [hi_fdims, pm_dims(d-1)];
        end
    end
    hca_inputs{l} = hi_input;
    hca_layer_names{l} = hi_layer_names;
    hca_fdims{l} = hi_fdims;
end

hca_input = {};
fdim = 0;

for l = 1:numel(opts.hieLayerName)
    if opts.kernelDegree > 1 && opts.homoKernel
        hi_output = ['l', num2str(l), '_', 'pm',num2str(opts.kernelDegree)];
        
        net.removeLayer(hca_layer_names{l}(1:end-1));
        
        ca_fdim = hca_fdims{l}(end);
    elseif opts.kernelDegree > 1
        % degree concatenation
        input = hca_inputs{l};
        hi_output = ['l', num2str(l), '_', 'dcat'];
        layer_name = ['layer', '_', num2str(l), '_', 'dconcat'];
        
        net.addLayer(layer_name, dagnn.Concat(), input, hi_output);
        
        ca_fdim = sum(hca_fdims{l});
    else
        hi_output = hca_inputs{1};
        ca_fdim = sum(hca_fdims{l});
    end
    
    hca_input = cat(2, hca_input, {hi_output});
    
    fdim = fdim + ca_fdim;
end

opts.fdim = fdim;

% layer concatenation
if numel(opts.hieLayerName) > 1
    hca_output = 'lcat';
    layer_name = 'layer_concat';
    net.addLayer(layer_name, dagnn.Concat(), hca_input, hca_output);
else
    hca_output = hca_input;
end

%% normalization layers
% ------------------------------------------------------------------------------------

% square-root
input = hca_output;
output = 'sqrt';
layer_name = 'sqrt';
net.addLayer(layer_name, SquareRoot(), input, output);

% l2
input = output;
output = 'l2norm';
layer_name = 'l2norm';
net.addLayer(layer_name, L2Norm(), input, output);

%% classification layer
% ------------------------------------------------------------------------------------
fc_params_init = getInitFCParams(net, imdb, opts);

input = output;
output = 'score';
layer_name = 'classifier';

fc_param_names = {'convclass_f', 'convclass_b'};

net.addLayer(layer_name, dagnn.Conv(), input, output, fc_param_names);

varId = net.getParamIndex(fc_param_names{1});
net.params(varId).value = fc_params_init{1};
net.params(varId).learningRate = 1;

varId = net.getParamIndex(fc_param_names{2});
net.params(varId).value = fc_params_init{2};
net.params(varId).learningRate = 1;

% loss functions
net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'score','label'}, 'objective');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'score','label'}, 'top1error');
net.addLayer('top5e', dagnn.Loss('loss', 'topkerror'), {'score','label'}, 'top5error');


for name = dagFindLayersOfType(net, 'dagnn.Conv')
    l = net.getLayerIndex(char(name)) ;
    net.layers(l).block.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
end


function layers = dagFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = [] ;
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
        layers{1,end+1} = net.layers(l).name ;
    end
end