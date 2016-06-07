% Stanford CS231A Final Project
% Due: 6/6/2016
% Author: G.K.
function [WARP_MATRIX] = GetWarpMatrix(motion_field)
% Find the warping matrix W such that:
% Im2 = W*Im1
% I.e. for a given color channel, the flattened Im2 vector is a linear
% combination of the intensities of the original Im1. The elements of W
% represent the weights of the points used for interpolating a given
% pixel. Because the interpolation is local, very few points are
% used in calculating the value of an interrogated point, so the matrix is
% sparse.

% This matrix W is required for optimization steps because it allows you to
% write the warped image as a matrix product: I2 = W*I1. This means you
% have a linear equation in I1.

% INPUTS:
% motion_field - H x W x 2 array of dense motion field. motion_field(:,:,1)
% is x direction, motion_field(:,:,2) for y direction.
% Is the motion field that take reference image to index i image:
% imI = reference + motion


[H,W,~] = size(motion_field);


%Meshgrid of pixel coordinates of reference image
[xr,yr] = meshgrid(1:W,1:H);
Npix = H*W;


%Coordinates after backtracing along vector field into index i image
x = reshape(xr + motion_field(:,:,1), Npix, 1);
y = reshape(yr + motion_field(:,:,2), Npix, 1);

%Corners around each pixel for interpolation 
x1 = floor(x); %LEFT
x2 = ceil(x); %RIGHT
y1 = floor(y); %TOP
y2 = ceil(y); %BOTTOM


%Some pixels are out of bounds:
%The pixes in the index i image are 0's since they don't have a
%corresponding place in the reference image. This is visible in the aligned
%background images in the OUTPUT folder.
%Ignore those pixels (they'll be left as zeros i nthe warping matrix, which
%is appropriate.

invalid_inds = (x1<1) | (y1<1) | (x2>W) | (y2>H);
valid_inds = ~invalid_inds;

x1 = x1(valid_inds);
x2 = x2(valid_inds);
y1 = y1(valid_inds);
y2 = y2(valid_inds);
x = x(valid_inds);
y = y(valid_inds);


%See https://en.wikipedia.org/wiki/Bilinear_interpolation
%which expresses f(x,y) as linear combination of 4 neighboring points. The
%weights are functions of only the distances of the point to the 4 nearby 
%sample points.
%WEIGHTS:
w11 = (x2-x).*(y2-y); %Top left
w21 = (y-y1).*(x2-x); %lower left
w12 = (y2-y).*(x-x1); %top right
w22 = (y-y1).*(x-x1); %lower right
%sm=w11+w21+w12+w22;
%sm(1:20)


%For each sampled point, get the 4 pixels from image index i that are
%involved in the calculation for that pixel. The flattened indices will be:
%top left corner, 
%top left corner + 1, %(the lower left corner since indexing runs down)
%top left corner + H, %(since indexing wraps aroudn and starts from top again)
%top left corner + H + 1
TLC_ind = sub2ind([H,W],y1,x1);
LLC_ind = TLC_ind + 1;
TRC_ind = TLC_ind + H;
LRC_ind = TRC_ind + 1;



%Make the row indices for the sparse matrix. Since every valid sample point
%has 4 associated interpolation corners, repeat 4 times (repeat each row
%index 4 times since there are 4 associated column indices):
row_inds = (1:Npix);
row_inds = row_inds(valid_inds);
I = vertcat(row_inds,row_inds,row_inds,row_inds)';
I = reshape(I,numel(I),1);

%Make the column indices
J = vertcat(TLC_ind, LLC_ind, TRC_ind, LRC_ind)';
J = reshape(J,numel(J),1);

%The actual interpolation weight values:
vals = reshape(horzcat(w11,w21,w12,w22),numel(w11)*4,1);


%Basic Diagnostics
%{
size(I)
size(J)
size(vals)
max(I(:))
max(J(:))
%}


%Create the sparse warping matrix
WARP_MATRIX = sparse(I,J,vals,Npix,Npix);

end