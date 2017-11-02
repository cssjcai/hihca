function fc_params_init = getInitFCParams(net, imdb, opts)

switch opts.pretrainFC
    case 'lr'
        
        if exist(fullfile(opts.modelTrainDir, 'fc_lr_init.mat'))
            load(fullfile(opts.modelTrainDir, 'fc_lr_init.mat')) ;
        else
            train = find(ismember(imdb.images.set, [1 2]));
            if ~exist(opts.initPolyfeatDir)
                mkdir(opts.initPolyfeatDir)
            end
            
            initPolyfeatList = dir(fullfile(opts.initPolyfeatDir, '/*.mat'));
            
            if numel(initPolyfeatList) == 0
                
                meta = net.meta;
                meta.augmentation.transformation = 'none';
                meta.augmentation.rgbVariance = [];
                meta.augmentation.numAugments = 1;
                
                batchSizeFC = floor(opts.batchSizeFC/meta.augmentation.numAugments);
                
                fn_trainFC = getBatchFn(opts, meta);
                
                for t=1:opts.batchSizeFC:numel(train)
                    fprintf('Initialization: extracting polynomial features of batch %d/%d\n', ceil(t/batchSizeFC), ceil(numel(train)/batchSizeFC));
                    batch = train(t:min(numel(train), t+batchSizeFC-1));
                    inputs = fn_trainFC(imdb, batch);
                    if opts.prefetch
                        nextBatch = train(t+batchSizeFC:min(t+2*batchSizeFC-1, numel(train)));
                        fn_trainFC(imdb, nextBatch);
                    end
                    
                    if opts.gpus
                        net.move('gpu');
                    end
                    
                    net.mode = 'test';
                    % net.conserveMemory = false;
                    net.eval(inputs(1:2));
                    fIdx = net.getVarIndex('l2norm');
                    polyFea = net.vars(fIdx).value;
                    polyFea = squeeze(gather(polyFea));
                    
                    for i=1:numel(batch)
                        fea_p = polyFea(:,i);
                        savefast(fullfile(opts.initPolyfeatDir, ['init_polyfeats_', num2str(batch(i), '%05d')]), 'fea_p');
                    end
                end
                
                if opts.gpus
                    net.move('cpu');
                end
            end
            
            
            % lr learning
            polydb = imdb;
            tempStr = sprintf('%05d\t', train);
            tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
            polydb.images.name = strcat('init_polyfeats_', tempStr{1}');
            polydb.images.id = polydb.images.id(train);
            polydb.images.label = polydb.images.label(train);
            polydb.images.set = polydb.images.set(train);
            polydb.imageDir = opts.initPolyfeatDir;
            
            [trainFV, trainY, valFV, valY] = load_polyfea_fromdisk(polydb);
            [w, b, acc, map, scores]= train_test_vlfeat('LR', trainFV, trainY, valFV, valY);
            
            % reshape the parameters to the input format
            fc_params_init{1} = reshape(single(w), 1, 1, size(w, 1), size(w, 2));
            fc_params_init{2} = single(squeeze(b));
            
            save(fullfile(opts.modelTrainDir, 'fc_lr_init.mat'), 'fc_params_init', '-v7.3');
        end
        
    case 'random'
        if exist(fullfile(opts.modelTrainDir, 'fc_ram_init.mat'))
            load(fullfile(opts.modelTrainDir, 'fc_ram_init.mat'));
        else
            numClass = length(unique(imdb.images.label));
            scal = 1.0;
            init_bias = 0.1;
            
            fc_params_init{1} = single(0.001/scal *randn(1, 1, opts.fdim, numClass));
            fc_params_init{2} = single(init_bias.*ones(1, numClass));
            
            save(fullfile(opts.modelTrainDir, 'fc_ram_init.mat'), 'fc_params_init', '-v7.3');
        end
end


function [trainFV, trainY, valFV, valY] = load_polyfea_fromdisk(polydb)
% -------------------------------------------------------------------------
train = find(polydb.images.set==1);
trainFV = cell(1, numel(train));
for i = 1:numel(train)
    load(fullfile(polydb.imageDir, polydb.images.name{train(i)}));
    trainFV{i} = fea_p;
end
trainFV = cat(2,trainFV{:});
trainY = polydb.images.label(train)';

val = find(polydb.images.set==2);
valFV = cell(1, numel(val));
for i = 1:numel(val)
    load(fullfile(polydb.imageDir, polydb.images.name{val(i)}));
    valFV{i} = fea_p;
end
valFV = cat(2,valFV{:});
valY = polydb.images.label(val)';
