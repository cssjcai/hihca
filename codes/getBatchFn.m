function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;
bopts.numAugments = meta.augmentation.numAugments;

fn = @(imdb,batch) getDagNNBatch(bopts,useGpu,imdb,batch) ;

function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
%isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

% if ~isVal
%   % training
%   im = cnn_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0) ;
% else
%   % validation: disable data augmentation
%   im = cnn_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0, ...
%                               'transformation', 'none') ;
% end

im = cnn_imagenet_get_batch(images, opts);
labels = imdb.images.label(batch) ;

labels = reshape(repmat(labels, opts.numAugments, 1), 1, size(im,4));

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', labels} ;
end