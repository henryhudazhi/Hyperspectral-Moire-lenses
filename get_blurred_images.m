%%get blurred images
clear all
clc

%load psf
load('psf_kmeasure1.mat')

% load hyper data
load('\reconstruction_results\new\460\cave\1.mat')

for i =1:25;

        Ir(:,:,i) = imfilter(gt(:,:,i), double(psf(:,:,i,1,1)), 'symmetric', 'conv');
   
        Ig(:,:,i) = imfilter(gt(:,:,i), double(psf(:,:,i,2,1)), 'symmetric', 'conv');
   
        Ib(:,:,i) = imfilter(gt(:,:,i), double(psf(:,:,i,3,1)), 'symmetric', 'conv');
        I = cat(3, Ir(:,:,i),Ig(:,:,i),Ib(:,:,i));
%         name = strcat(num2str(i),'.bmp')
%         imwrite(I,name)
end
yr = sum(Ir, 3);
% figure()
% imshow(yr,[])

yg = sum(Ig, 3);
% figure()
% imshow(yg,[])

yb = sum(Ib, 3);
% figure()
% imshow(yb,[])

y = cat(3, yr,yg,yb);
% figure()
% imshow(y,[])
imwrite(y,'kaist2.bmp')
    
% outname = strcat('test1.mat');
% save(outname,'y');