function [imdb] = data_loader(opts)

if exist(opts.datasetImdbDir)
    imdb = load(opts.datasetImdbDir) ;
    return ;
end

switch opts.dataset
    case 'cubcrop'
        imdb = cub_get_database(opts.datasetDir, true, false);
    case 'cub'
        imdb = cub_get_database(opts.datasetDir, false, false);
    case 'aircraft'
        imdb = aircraft_get_database(opts.datasetDir, 'variant');
    case 'cars'
        imdb = cars_get_database(opts.datasetDir, false, false);
    otherwise
        error('Unknown dataset %s', opts.dataset);
end

save(opts.datasetImdbDir, '-struct', 'imdb') ;
