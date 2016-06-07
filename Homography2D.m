function [H] = Homography2D(x1,x2)
% Normalized DLT (Direct Linear Transform) for 2D Homography:
% Given a set of coordinates in image1 and image2, find the homography, H,
% that maps points in image1 to points in image2.
% x1 and x2 are Nx3 matrices of homogeneous coodinates of corresponding pts

%x1 and x2 should have same dimensions
N = size(x1,1);


% NORMALIZATION:
%--------------------------------------------------------------------------
% Do a normalizing transformation for each set of image points:
% x' = Tx
% where T is 3x3 transformation matricxthat translates and scales the 
% coordinates to be 0 mean, with average distance from origin sqrt(2)

eps = .000000001;

% Image1
mu1 = mean(x1);
dx1 = x1(:,1) - mu1(1);
dy1 = x1(:,2) - mu1(2);
d_ave1 = sum(sqrt(dx1.^2 + dy1.^2))/N;
K1 = sqrt(2)/(d_ave1+eps);

% Image2
mu2 = mean(x2);
dx2 = x2(:,1) - mu2(1);
dy2 = x2(:,2) - mu2(2);
d_ave2 = sum(sqrt(dx2.^2 + dy2.^2))/N;
K2 = sqrt(2)/(d_ave2+eps);

% Make the transformation matrices
T1 = [[K1, 0, -K1*mu1(1)]; [0, K1, -K1*mu1(2)]; [0, 0, 1]];
T2 = [[K2, 0, -K2*mu2(1)]; [0, K2, -K2*mu2(2)]; [0, 0, 1]];

% Apply the normalization transformations to image1 and image2
x1_t = (T1*x1.').'; %is column vectors of [x, y, 1]
x2_t = (T2*x2.').'; %is column vectors of [x, y, 1]
%--------------------------------------------------------------------------




% BUILDING MATRICES AND SOLVING SYSTEM OF EQUATIONS:
%--------------------------------------------------------------------------
% Build a 2N x 9 matrix representing constraints from each of the N points
top_half = horzcat(zeros(N,3), -x1_t, repmat(x2_t(:,2),1,3).*x1_t);
bottom_half = horzcat(x1_t, zeros(N,3), -repmat(x2_t(:,1),1,3).*x1_t);
A = vertcat(top_half,bottom_half);

% Solve Ah = b for h vector (homography parameters)
[U,S,V] = svd(A);
h9 = V(:,end);
H_t = reshape(h9,3,3).';

% Since normalization was used before, now denormalize it
H = T2\(H_t*T1); %(By design, know that T1 is invertible)
%--------------------------------------------------------------------------