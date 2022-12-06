function [ ] = generate_dataset_mixture( datadir, fns, newdir, sigmas, gt_key, preprocess )
%GENERATE_DATASET Summary of this function goes here    
    ratios = [0.1, 0.3, 0.5, 0.7];
    min_amount = 0.05;
    max_amount = 0.15;
    if ~exist(newdir, 'dir')
        mkdir(newdir)
    end
        
    for k = 1:length(fns)        
        fn = fns{k};
        fprintf('generate data(%d/%d)\n', k, length(fns));
        filepath = fullfile(datadir, fn);
        mat = load(filepath); % contain gt_key
        gt = getfield(mat, gt_key);
        
        if exist('preprocess', 'var')
            gt = preprocess(gt);
        end
        
        gt = normalized(gt);
        % sample sigma uniformly from sigmas
        idx = randi(length(sigmas), size(gt,3), 1);
        sigma = sigmas(idx);
        sigma = sigma(1);
        s = sigma*rand(1,size(gt,3));
        s = reshape(s, 1, 1, length(s));
        input = gt + s/255 .* randn(size(gt));

        [~, N, B] = size(gt);
        
        % add stripe
        all_band = randperm(B);
        b = floor(B/3);
        band_stripe = all_band(1:b);
                
        stripnum = randi([ceil(min_amount * N), ceil(max_amount * N)], length(band_stripe), 1);
        fprintf('Stripes:\n');
        disp(stripnum);
        for i=1:length(band_stripe)
            loc = randperm(N);
            loc = loc(1:stripnum(i));
            stripe = rand(1,length(loc))*0.5-0.25;
            input(:,loc,band_stripe(i)) = input(:,loc,band_stripe(i)) - stripe;
        end
        
        % add deadline
        band_deadline = all_band(b+1:2*b);
        deadlinenum = randi([ceil(min_amount * N), ceil(max_amount * N)], length(band_deadline), 1);
        fprintf('Deadline:\n');
        disp(deadlinenum);
        for i=1:length(band_deadline)
            loc = randperm(N);
            loc = loc(1:deadlinenum(i));
            input(:,loc,band_deadline(i)) = 0;
        end
        
        % add impulse        
        fprintf('impulse:\n');
        band_impulse  = all_band(2*b+1:3*b);     
        idx = randi(length(ratios), length(band_impulse), 1);
        ratio = ratios(idx);
        disp(ratio);
        for i=1:length(band_impulse)
            input(:,:,band_impulse(i)) = imnoise(input(:,:,band_impulse(i)),'salt & pepper',ratio(i));
        end
        DataCube = input;
        save(fullfile(newdir, fn), 'DataCube','s','band_stripe','stripnum','band_deadline','deadlinenum','band_impulse','ratio');
    end
end


function gt=normalized(gt)
    gt=gt./max(gt(:));
end
