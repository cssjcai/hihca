function [opts] = hihca_setup(varargin)

opts.runPhase = 'test';
opts.gpus = [1];
opts.cnnModel = 'vgg16';
opts.cnnModelDir = '../models/pretrained_models/imagenet-vgg-verydeep-16.mat';
opts.dataset = 'cub';
opts.datasetDir = '../datasets';

opts.imageScale = 2;
opts.hieLayerName = {'relu5_3'};
opts.layerFusion = 'hc';
opts.rescaleLayerFactor = [1];
opts.kernelDegree = 2;
opts.homoKernel = true;
opts.num1x1Filter = [8192];
opts.init1x1Filter = 'rad';
opts.pretrainFC = 'lr';
opts.batchSizeFC = 64;

opts.numEpochs = 50;
opts.batchSize = 16;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005;
opts.momentum = 0.9;
opts.prefetch = false ;
opts.cudnnWorkspaceLimit = 1024*1024*1204; 
opts.numFetchThreads = 12;

[opts, varargin] = vl_argparse(opts, varargin);

opts.modelIDX = [opts.dataset, '_', opts.cnnModel, '_', '{', opts.hieLayerName{:}, '}',...
                  '_', opts.layerFusion, '_', 'kd', num2str(opts.kernelDegree), '_', 'homo', num2str(opts.homoKernel)];

opts.modelTrainDir = fullfile('../models', opts.modelIDX);
opts.datasetImdbDir = fullfile(opts.modelTrainDir, 'imdb.mat');
opts.init1x1FilterDir = fullfile(opts.modelTrainDir, 'init_1x1filters');
opts.initPolyfeatDir = fullfile(opts.modelTrainDir, 'init_polyfeats');
opts.trainedPolyfeatDir = fullfile(opts.modelTrainDir, 'trained_polyfeats');
opts.trainedFeaturemapDir = fullfile(opts.modelTrainDir, 'trained_featuremaps');

opts = vl_argparse(opts, varargin);

if(~exist(opts.modelTrainDir, 'dir'))
    mkdir(opts.modelTrainDir);
end
