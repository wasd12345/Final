%This warping function is from a utility function used in 
%Ce Liu's Optical Flow package from:
%https://people.csail.mit.edu/celiu/OpticalFlow/
function warpI2=WARP(im1,im2,vx,vy)
if isfloat(im1)~=1
    im1=im2double(im1);
end
if isfloat(im2)~=1
    im2=im2double(im2);
end

for i=1:3
    [im,isNan]=warpFL(im2(:,:,i),vx,vy);
    temp=im1(:,:,i);
    warpI2(:,:,i)=im;
end