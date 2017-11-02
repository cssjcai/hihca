function hihca_train(imdb, opts)

%% build hihca network
% ------------------------------------------------------------------------------------
net = hihca_model(imdb, opts);

%% train hihca network
% ------------------------------------------------------------------------------------
opts.train.numEpochs = opts.numEpochs;
opts.train.batchSize = opts.batchSize;
opts.train.learningRate = opts.learningRate;
opts.train.weightDecay = opts.weightDecay;
opts.train.momentum = opts.momentum;
opts.train.numSubBatches = 1;
opts.train.continue = true;
opts.train.gpus = opts.gpus;
opts.train.expDir = opts.modelTrainDir;

meta = net.meta;
meta.augmentation.transformation = 'f2';
meta.augmentation.rgbVariance = [];
meta.augmentation.numAugments = 1;

fn_train = getBatchFn(opts, meta);

[net, info] = cnn_train_dag(net, imdb, fn_train, opts.train);

% net = cnn_imagenet_deploy(net);
% save(fullfile(opts.modelTrainDir, 'trained-model.mat'), 'net', 'info', '-v7.3');
