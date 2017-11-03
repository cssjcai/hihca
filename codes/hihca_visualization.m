function hihca_visualization(imdb, opts)
% using homogeneous kernel for better visualization of higher-order part
% with specific degree

epochIDX = ['net-epoch-',num2str(opts.netIDX),'.mat'];
snet = load(fullfile(opts.modelTrainDir, epochIDX), 'net', 'stats');
net = dagnn.DagNN.loadobj(snet.net);
net.removeLayer({'sqrt','l2norm','classifier','loss','error','top5e'});

U = cell(1,opts.kernelDegree);

for d = 1:opts.kernelDegree
    pIDX = net.getParamIndex(['conv_d',num2str(opts.kernelDegree),'fm',num2str(d)]);
    U{d} = net.params(pIDX).value;
end

convIDX = net.getLayerIndex(opts.hieLayerName{end});

lnames_r = dagFindRemovedLayerNames(net, convIDX);

net.removeLayer(lnames_r);

if ~exist(opts.trainedFeaturemapDir)
    mkdir(opts.trainedFeaturemapDir)
end

if exist(fullfile(opts.modelTrainDir, 'fc_svm_trained.mat'))
    load(fullfile(opts.modelTrainDir, 'fc_svm_trained.mat'));
    w = fc_params_trained{1};
else
    fprintf('You need to run testing first then visualization !\n');
    return
end

vis_set = find(ismember(imdb.images.set, [1 2]));
vis_classIDX = [1]; % vis_classIDX = [1:size(w,2)];
vis_topK = 3;
for c = vis_classIDX
    
    trainedFeaturemapClassDir = fullfile(opts.trainedFeaturemapDir, imdb.classes.name{c});
    
    if ~exist(trainedFeaturemapClassDir)
        mkdir(trainedFeaturemapClassDir)
    end

    [~, wIDX] = sort(abs(w(:,c)),'descend');    
    wIDX_topK = wIDX(1:vis_topK); 
    
    chIDX = cell(1, vis_topK);
    for i = 1:vis_topK
        uuT = U{1}(:,wIDX_topK(i))*U{2}(:,wIDX_topK(i))';
        if opts.kernelDegree > 2
            for d = 3:opts.kernelDegree
                uuT = kron(uuT,U{d}(:,wIDX_topK(i))');
            end
            uuT = reshape(uuT, size(U{1},1)*ones(1, opts.kernelDegree));
        end
        ind = find(uuT==max(uuT(:)));
        % for degree-2
        [ch1,ch2] = ind2sub(size(U{1},1)*ones(1,opts.kernelDegree), ind);
        chIDX{i} = [ch1,ch2];
        % for degree-3
        %[ch1,ch2,ch3] = ind2sub(size(U{1},1)*ones(1,opts.kernelDegree), ind);
        %chIDX{i} = [ch1,ch2,ch3];
    end    
    
    vis_set_c = vis_set(imdb.images.label==c);
    
    meta = net.meta;
    meta.augmentation.transformation = 'none' ;
    meta.augmentation.rgbVariance = [];
    meta.augmentation.numAugments = 1;
    
    batchSizeFC = floor(opts.batchSizeFC / 4*meta.augmentation.numAugments);
    
    fn_trainFC = getBatchFn(opts, meta) ;
    
    for t=1:batchSizeFC:numel(vis_set_c)
        fprintf(['Visualizating for ', imdb.classes.name{c}, ' : extracting feature maps of batch',' %d/%d\n'], ceil(t/batchSizeFC), ceil(numel(vis_set_c)/batchSizeFC));
        batch = vis_set_c(t:min(numel(vis_set_c), t+batchSizeFC-1));
        inputs = fn_trainFC(imdb, batch) ;    
        
        if opts.gpus
            net.move('gpu') ;
        end
        
        net.mode = 'test' ;
        net.conserveMemory = false;
        net.eval(inputs(1:2));
        
        hcaMaps = [];
        for l = 1:numel(opts.hieLayerName)
            lIdx = net.getLayerIndex(opts.hieLayerName(l));
            fIdx = net.getVarIndex(net.layers(lIdx).outputs);
            caMaps = net.vars(fIdx).value;
            caMaps = squeeze(gather(caMaps));
            hcaMaps = cat(3, hcaMaps, caMaps);
        end
        
        for i=1:numel(batch)
            
            image_name = {fullfile(imdb.imageDir,imdb.images.name{batch(i)})};           
            vopts.imageSize = meta.normalization.imageSize ;
            vopts.border = meta.normalization.border;
            
            image = cnn_imagenet_get_batch(image_name, vopts);
            image = double(image)/255;
            for id = 1:numel(chIDX)
                chMaps = hcaMaps(:,:,chIDX{id},i);
                image_partMap_all = [];
                for p = 1:numel(chIDX{id})
                    partMap = chMaps(:,:,p);
                    [X,Y] = meshgrid(1:size(partMap,2), 1:size(partMap,1));
                    [Xq,Yq] = meshgrid(1:0.01:size(partMap,2), 1:0.01:size(partMap,1));
                    partMap_ = interp2(X, Y, partMap, Xq, Yq, 'linear');
                    partMap_ = imresize(partMap_, net.meta.normalization.imageSize(1:2));
                    partMap_nor = partMap_-min(partMap_(:));
                    partMap_nor = (partMap_nor)./max(partMap_nor(:));
                    
                    partMap_nor = mat2gray(partMap_nor);
                    partMap_nor_gray = gray2ind(partMap_nor,256);
                    partMap_nor_rgb = ind2rgb(partMap_nor_gray, jet(256));
                    
                    image_partMap = image*0.4 + partMap_nor_rgb*0.5;                    
                    image_partMap = padarray(image_partMap,[20 20],255,'both');
                    
                    image_partMap_all = cat(2, image_partMap_all, image_partMap);
                end
                % imshow(image_partMap_all)
                imwrite(image_partMap_all, fullfile(trainedFeaturemapClassDir, [num2str(batch(i), '%05d'),'_', 'trained_featuremaps_', num2str(id, '%02d'), '.jpg']));
            end           
        end
        
    end
       
end


function layernames = dagFindRemovedLayerNames(net, convIDX)
% -------------------------------------------------------------------------
layernames = {} ;
for l = 1:numel(net.layers(convIDX+1:end))
    layernames = cat(2, layernames, {net.layers(convIDX+l).name});
end