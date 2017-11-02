function pnet = hihca_model(imdb, opts)

%% load the pre-trained cnn model as base net
% ------------------------------------------------------------------------------------
bnet = load(opts.cnnModelDir);

tru_idx = simpleFindLayerIDXOfName(bnet, opts.hieLayerName{end});
bnet.layers = bnet.layers(1:tru_idx);
bnet = vl_simplenn_tidy(bnet);

bnet_info = vl_simplenn_display(bnet);

hieLayerIDX = [];
for l = 1:numel(opts.hieLayerName)
    hieLayerIDX = [hieLayerIDX, simpleFindLayerIDXOfName(bnet, opts.hieLayerName{l})];
end

opts.whc = bnet_info.dataSize([1 2 3], hieLayerIDX+1);

%% transform simplenn to dagnn
% ------------------------------------------------------------------------------------
pnet = dagnn.DagNN();
pnet = pnet.fromSimpleNN(bnet, 'CanonicalNames', true);

opts.hieLayerIDX = [];
for l = 1:numel(opts.hieLayerName)
    opts.hieLayerIDX = [opts.hieLayerIDX, pnet.getLayerIndex(opts.hieLayerName{l})];
end

pnet.meta.classes.name = imdb.classes.name;
pnet.meta.classes.description = imdb.classes.name;

pnet.meta.normalization.imageSize = [pnet.meta.normalization.imageSize(1:2)*opts.imageScale, pnet.meta.normalization.imageSize(3)];
pnet.meta.normalization.averageImage = imresize(pnet.meta.normalization.averageImage, opts.imageScale);

clear bnet;

%% build hihca net
% ------------------------------------------------------------------------------------
switch opts.layerFusion
    case 'hc'
        pnet = hihca_model_hc(pnet, imdb, opts);
    case 'hed'
        pnet = hihca_model_hed(pnet, imdb, opts);
    otherwise
        error('Unknown layer integration!') ;
end


function layers = simpleFindLayerIDXOfName(net, name)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.name, name), net.layers)) ;
