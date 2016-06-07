% Stanford CS231A Final Project
% Due: 6/6/2016
% Author: G.K.

%This code is to demonstrate that GetWarpMatrix function works as intended.
%This code is not actually called in the MAIN script, but is important to
%confirm that everything is working as expected.


%Test GetWarpMatrix function:

%Load an example image
im_ref = imread(fullfile(pwd,'INPUT vending4','20160527_151541.jpg'));
im_ref = im_ref(1:400,1:500,:);
[H,W,C] = size(im_ref)

im_ref = im2double(im_ref);

%Make very simple motion field: just translation
motion_field = 17.2*ones(H,W,2);
size(motion_field)
WARP_MATRIX = GetWarpMatrix(motion_field);




%Make flattened vector of image
warped = zeros(size(im_ref));
for c=1:3
    flat = reshape(im_ref(:,:,c),H*W,1);
    warpIMflat = WARP_MATRIX*flat; %since same matrix for all 3 color channels
    warped(:,:,c) = reshape(warpIMflat,H,W);
end



%Compare original to warped:
figure; imshow(im_ref), title('Original Image')
figure; imshow(warped), title('Warping Matrix on Image') %This image really is the bilinear interpolation



%Get the built-in bilinear interpolation image as comparison
'Getting MATLAB built-in bilinear interpolation...'
[X, Y] = meshgrid((1:W),(1:H));
xi = X + motion_field(:,:,1);
yi = Y + motion_field(:,:,2);

X = reshape(X,numel(X),1);
Y = reshape(Y,numel(Y),1);

Z = reshape(im_ref(:,:,1),H*W,1);
bilinear_R = scatteredInterpolant(X,Y,Z,'linear','none');
bilinear_R = bilinear_R(xi,yi);

Z = reshape(im_ref(:,:,2),H*W,1);
bilinear_G = scatteredInterpolant(X,Y,Z,'linear','none');
bilinear_G = bilinear_G(xi,yi);

Z = reshape(im_ref(:,:,3),H*W,1);
bilinear_B = scatteredInterpolant(X,Y,Z,'linear','none');
bilinear_B = bilinear_B(xi,yi);

bilinear = cat(3,bilinear_R,bilinear_G,bilinear_B);

%View the MATLAB bilinear interpolation result:
figure; imshow(bilinear), title('Built-in Bilinear Interpolated Image')

%The images look virtually identical, indiciating that the matrix form of 
%bilinear interpolation used in GetWarpMatrix works as intended

%The exact values are slightly different though (e.g. after 3rd decimal)
%This is likely because MATLAB has some additional steps that make minor
%adjustments. Either way, this is very close and is good enough for the
%optimization steps.
%{
format long;
bilinear(30:40,30:40,1)
warped(30:40,30:40,1)
bilinear(30:40,30:40,1)-warped(30:40,30:40,1)
%}
