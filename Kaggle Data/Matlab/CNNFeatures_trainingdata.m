%Computes the Feature Vectors for training images
%Stores the feature vectors in 2 files
%FV_unnorm.mat stores the unnormalized feature vectors for training images 
%FV_norm.mat stores the normalized feature vectors for training images
%size of each = #Images,1,4096

% load the VGG16 network

net = load('imagenet-vgg-verydeep-16.mat') ;
net = vl_simplenn_tidy(net) ;

% Load the training images

base = '../train_photos/';
load('../data_pickle/images_train.mat');

FV_unnorm = [];
FV_norm = [];

for num = 1:size(images_train,2)
    name = strcat(base, sprintf('%d',images_train(1,num)), '.jpg');
    disp(name);
    disp(num);
    im = imread(name);
    im = im2single(im);
    height = size(im,1);
    width = size(im,2);
    
    if(height<width)
        im = imresize(im, [256, NaN]);
    end
    
    if(width<=height)
        im = imresize(im, [NaN, 256]);
    end
    
    %repr = single(zeros(1,1,4096));
    height = size(im,1);
    width = size(im,2);   
    im = im(floor(height/2)-112:floor(height/2)+111, floor(width/2)-112:floor(width/2)+111, :);
    
    disp(size(im));
    
    res = vl_simplenn(net, im);
    repr = res(36).x;
   	
    FV_unnorm = cat(1,FV_unnorm, repr);
    
    rho = 5;
    kappa = 0;
    alpha = 1;
    beta = 0.5;
    y_nrm = vl_nnnormalize(repr, [rho kappa alpha beta]); %stores the L2 normalised feature vector 
    
    FV_norm = cat(1,FV_norm, y_nrm);   
    
end

    save('FV_unnorm','FV_unnorm');
    save('FV_norm','FV_norm');


