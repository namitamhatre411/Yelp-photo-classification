%Computes the Feature Vectors for test images
%Stores the feature vectors in 2 files
%FV_unnorm_test.mat stores the unnormalized feature vectors for test images 
%FV_norm_test.mat stores the normalized feature vectors for test images 
%size of each = #Images,1,4096

% load the VGG16 network

net = load('imagenet-vgg-verydeep-16.mat') ;
net = vl_simplenn_tidy(net) ;

% Load the training images

base = '../train_photos/';
load('../data_pickle/images_test.mat');

FV_unnorm_test = [];
FV_norm_test = [];

for num = 1:size(images_test,2)
    name = strcat(base, sprintf('%d',images_test(1,num)), '.jpg');
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
   	
    FV_unnorm_test = cat(1,FV_unnorm_test, repr);
    
    rho = 5;
    kappa = 0;
    alpha = 1;
    beta = 0.5;
    y_nrm = vl_nnnormalize(repr, [rho kappa alpha beta]); %stores the L2 normalised feature vector 
    
    FV_norm_test = cat(1,FV_norm_test, y_nrm);   
    
end

    save('FV_unnorm_test','FV_unnorm_test');
    save('FV_norm_test','FV_norm_test');


