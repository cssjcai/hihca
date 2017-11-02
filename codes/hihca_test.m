function hihca_test(imdb, opts)

epochidx = ['net-epoch-',num2str(opts.netIDX),'.mat'];
snet = load(fullfile(opts.modelTrainDir, epochidx), 'net', 'stats');
net = dagnn.DagNN.loadobj(snet.net);
net.removeLayer({'classifier','loss','error','top5e'});

train = find(ismember(imdb.images.set, [1 2]));
if ~exist(opts.trainedPolyfeatDir)
    mkdir(opts.trainedPolyfeatDir)
end

trainedPolyfeatList = dir(fullfile(opts.trainedPolyfeatDir, '/*.mat'));

if numel(trainedPolyfeatList) == 0
    
    meta = net.meta;
    meta.augmentation.transformation = 'f2';
    meta.augmentation.rgbVariance = [];
    meta.augmentation.numAugments = 2;
    
    batchSizeFC = floor(opts.batchSizeFC / meta.augmentation.numAugments);
    
    fn_trainFC = getBatchFn(opts, meta);
    
    for t=1:batchSizeFC:numel(train)
        fprintf('Testing: extracting polynomial features of batch %d/%d\n', ceil(t/batchSizeFC), ceil(numel(train)/batchSizeFC));
        batch = train(t:min(numel(train), t+batchSizeFC-1));
        inputs = fn_trainFC(imdb, batch);       
        
        if opts.gpus
            net.move('gpu');
        end
        
        net.mode = 'test' ;
        net.eval(inputs(1:2));
        fIdx = net.getVarIndex('l2norm');
        polyFea = net.vars(fIdx).value;
        polyFea = squeeze(gather(polyFea));
        
        for i=1:numel(batch)
            fea_p = polyFea(:,meta.augmentation.numAugments*(i-1)+1:meta.augmentation.numAugments*i);
            savefast(fullfile(opts.trainedPolyfeatDir, ['trained_polyfeats_', num2str(batch(i), '%05d')]), 'fea_p');
        end
    end
    
    % move back to cpu
    if opts.gpus
        net.move('cpu');
    end
end


% lr learning
polydb = imdb;
tempStr = sprintf('%05d\t', train);
tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
polydb.images.name = strcat('trained_polyfeats_', tempStr{1}');
polydb.images.id = polydb.images.id(train);
polydb.images.label = polydb.images.label(train);
polydb.images.set = polydb.images.set(train);
polydb.imageDir = opts.trainedPolyfeatDir;
polydb.numAugments = 2;

[trainFV, trainY, valFV, valY] = load_polyfea_fromdisk(polydb);
[w, b, acc, map, scores]= train_test_vlfeat('SVM', trainFV, trainY, valFV, valY);

fc_params_trained{1} = w;
fc_params_trained{2} = b;

save(fullfile(opts.modelTrainDir, 'fc_svm_trained.mat'), 'fc_params_trained', '-v7.3') ;


function [trainFV, trainY, valFV, valY] = load_polyfea_fromdisk(polydb)
% -------------------------------------------------------------------------
train = find(polydb.images.set==1);
trainFV = cell(1, numel(train));
for i = 1:numel(train)
    load(fullfile(polydb.imageDir, polydb.images.name{train(i)}));
    trainFV{i} = fea_p;
end
trainFV = cat(2,trainFV{:});
%trainFV = trainFV(:,2:2:end);
trainY = polydb.images.label(train)';
trainY = reshape(repmat(trainY,1, polydb.numAugments)', [], 1);

val = find(polydb.images.set==2);
valFV = cell(1, numel(val));
for i = 1:numel(val)
    load(fullfile(polydb.imageDir, polydb.images.name{val(i)}));
    valFV{i} = fea_p;
end
valFV = cat(2,valFV{:});
valFV = valFV(:,1:2:end);
valY = polydb.images.label(val)';