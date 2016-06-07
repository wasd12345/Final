% Stanford CS231A Final Project
% Due: 6/6/2016
% Author: G.K.
% (repurposed from my previous CS231A HW assignment)
function [inliers, H] = RANSAC_Homography2D(P1,P2,pixelThresh)%(P1,P2)

N=length(P1);

%P1 is format:
% [[ x1, ..., xN],
%  [ y1, ..., yN],
%  [ 1, ...,  1 ]]
%And same for P2


%Parameters
iter=10000



% X is a 3 x N matrix where each col is the homogeneous coordinates of a
% point in image1, while Xp is the corresponding points in image2.

%Main loop, picks a set of 4 random points, calculates homography,
%calculates inliers.
N_in_best = 0;
for iteration=1:iter
       
    % Randomly choose 4 points (sample without replacement)
    r = randperm(N);
    inds4 = r(1:4);

    %Calculate model
    x1 = P1(:,inds4).';
    x2 = P2(:,inds4).';
    %x1 and x2 are 4x3 matrices of coordinates
    H = Homography2D(x1,x2);
    H = H/H(3,3);
    
    %Calculate reprojection error
    Xp_t = H*P1; %3xN matrix of im2 pts calculated by projecting im1 pts
    Xp_t = Xp_t./repmat(Xp_t(3,1:end),3,1);
    diff = (P2 - Xp_t);
    error_i = sqrt(sum(diff(1:2,1:end).^2));
    
    
    %Calculate inliers
    valid_pts = find(error_i < pixelThresh);
    N_in = length(valid_pts);
    
    %Keep track of best model
    if N_in > N_in_best
        inliers = valid_pts;
        
        %The homography itewlf doesn't actually matter: just using projective
        %transofmration as a way of separating the 2 layers. However, could
        %potentially improve separation by forcing the homography to have
        %realistic values. basically a way of regularization.
        %{
        %Recalculate model using ALL inliers:
        x1 = P1(:,inliers).';
        x2 = P2(:,inliers).';
        %x1 and x2 are Nx3 matrices of coordinates
        H = Homography2D(x1,x2);
        H = H/H(3,3);
        %}
        
        N_in_best = N_in;
    end
    
end
N_in_best
end

