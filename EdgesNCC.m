% Stanford CS231A Final Project
% Due: 6/6/2016
% Author: G.K.
function [X, Y, VX, VY, NCC] = EdgesNCC(im_ref,im,NCC_WINDOW_RADIUS,SUBPIX_WINDOW_RADIUS,RELATIVE_THRESH,ABSOLUTE_THRESH,NpixUB)
%-Iterate through all edge pixels of im
%-Find the NCC between window around edge pixel and reference im1
%-Find the highest peak (peak1) in cross correlation space
%-The peak1 coordinates give the motion for this edge pixel from im -> im_ref

%-If the highest peak is < ABSOLUTE_THRESH then flag this pixel as questionable
%-If the 2nd highest peak is > RELATIVE_THRESH*peak1 then flag this pixel as questionable


%INPUTS
%im and im_ref are binary edge maps (e.g. after doing Canny)


%The C++ mex code version of openCV NCC requires type uint8:
im_ref = im2uint8(im_ref);
im = im2uint8(im);
%figure;imshow(im2)


%Get indices of edge pixels in im2
%edge_inds = find(im2==1)
edge_inds = find(im==255.);%Now that using uint8



%Randomly choose ~10-20% of edge pixels because there are so many
Npix_total = floor(length(edge_inds)/5) %Even this is often very many pixels
edge_inds = datasample(edge_inds,Npix_total,'Replace',false);
size(edge_inds);
Npix = min(NpixUB,Npix_total)

%Container arrays of x and y indices, components of flow, NCC
X = [];
Y = [];
VX = [];
VY = [];
NCC = [];

%Iterate through each edge pixel, calculate the NCC for a window around
%that pixel, find the peak in NCC space, and save the offset as the 
%components of the motion vector for that pixel. 
for pix=1:Npix
    
    %Just to monitor progress, print every 50 pixels finished
    if mod(pix,50) == 0
        pix
    end
    
    [y2,x2] = ind2sub(size(im),edge_inds(pix));
       
    %Get window around pixel. If window would be out of bounds, ignore it.
    %Besides, pixels at sides of im might be less reliable since they may
    %not have counterparts in reference image, depending on direction of
    %motion.
    try
        TEMPLATE = im(y2-NCC_WINDOW_RADIUS:y2+NCC_WINDOW_RADIUS, x2-NCC_WINDOW_RADIUS:x2+NCC_WINDOW_RADIUS);
    catch
        continue
    end
    
    
    %NCC
    %C = normxcorr2(TEMPLATE,im_ref);%previously used builtin MATLAB code
    C = CCORR_NORM(TEMPLATE,im_ref); %using my openCV mex file %at least an order of magnitude faster
    ncc = max(C(:));
    %figure, surf(C), shading flat
    
    
    %ABSOLUTE THRESHOLD on NCC to exclude low scores:
    if ncc < ABSOLUTE_THRESH
        continue
    end
    
    %RELATIVE THRESHOLD: could check if 2nd highest peak in NCC 
    %space is > thresh*peak. If yes, then flag this pixel as possibly unreliable
    %or just continue and exclude it.
     
       
    
    %Peaks in NCC space
    [ypeak, xpeak] = find(C==max(C(:)));
    %Very rarely, but sometimes, this finds 2 exact same scores... 
    %... so take the 1st in case of a tie:
    ypeak = ypeak(1);
    xpeak = xpeak(1);
    %Possibly better ways to deal with this, e.g. pick randomly, or average
    %the two together, or exclude if the tied peaks are far apart.
    
    %Get subpixel accuracy by taking centroid of small window centered on
    %the peak: e.g. 3x3 or 5x5 window. If  the peak is very close to edge,
    %small window may be out of bounds, so ignore it and continue:
    try
        chip = C(ypeak-SUBPIX_WINDOW_RADIUS:ypeak+SUBPIX_WINDOW_RADIUS, xpeak-SUBPIX_WINDOW_RADIUS:xpeak+SUBPIX_WINDOW_RADIUS);
    catch
        continue
    end
    t = (-SUBPIX_WINDOW_RADIUS:SUBPIX_WINDOW_RADIUS);
    colsum = sum(chip);
    dx = dot(t,colsum)/sum(colsum);
    rowsum = sum(chip,2);
    dy = dot(t,rowsum)/sum(rowsum);
    
    %Get the translation going from im2 -> reference (accounting for
    %template shape):
    y_ref = ypeak + (NCC_WINDOW_RADIUS) + dy;
    x_ref = xpeak + (NCC_WINDOW_RADIUS) + dx;
    
    %Motion field components
    vx = x_ref-x2;
    vy = y_ref-y2;
    
    
    %DIRECTIONAL THRESHOLD:
    %Tell the user to take a roughly horizontal pan across the scene.
    %Therefore, flow should not have strong vertical vectors with Y 
    %components mcuh greater than X component: exclude if > than XYZ degs:
    %[could even be more strict and tell user to always go left -> right...]
    if abs(vy/vx) > .6 %.6 for ~30 degs %1. for 45 degs
        continue
    end
    
    
    %Append to arrays if none of above exclusion conditions met
    VX = cat(1,VX,vx);
    VY = cat(1,VY,vy);
    X = cat(1,X,x2);
    Y = cat(1,Y,y2);
    NCC = cat(1,NCC,ncc);
    
    
    %{
    figure
    hAx  = axes;
    imshow(im_ref,'Parent', hAx);
    %format is [top left X, top left Y, width, height]
    imrect(hAx, [xoffSet+1, yoffSet+1, size(TEMPLATE,2), size(TEMPLATE,1)]);
    
    figure;imshow(TEMPLATE);
    figure;imshow(im_ref);
    
    y1 = yoffSet-window_radius;
    y2 = yoffSet+window_radius;
    x1 = xoffSet-window_radius;
    x2 = xoffSet+window_radius;    
    figure;imshow(im_ref(y1:y2,x1:x2));
    %}
    
end


%There will likely be some bad flow vectors from spurious NCC matches.
%RANSAC will mostly take care of this, but we can improve the usefulness of
%RANSAC by increasing the inlier to outlier ratio. Do this by getting rid
%of flow vectors with very obviously wrong angles and magnitudes.
%This is basically just a form of regularized NCC: I'm smoothing out the
%flow so the vectors are more uniform and don't overfit the data at
%spurious matches.
angles  = (180./pi)*atan2(VY,VX);
magnitudes  = sqrt(VX.^2 + VY.^2);

%Find the median magnitude and angle, and get rid of vectors that are
%significantly different. E.g. any "real" motion vector should have
%magnitude within ~ +/- 5 pixels of "true" magnitude [have to keep in mind that 
%depending on depth, the magnitude will be different, but even being
%generous to account for this will be helpful to get rid of obviously bad
%flow vectors]. Also, since the user is told to pan horizontally across the 
%view, all background and foreground vectors should have close to the same
%horizontal direction. This allows to further prune out bad flow vectors to
%increase success with RANSAC.

median_distance_thresh = 10; %in pixels. Exclude vectors w/ magnitudes > this many pixels from median magnitude.
M = abs(magnitudes - median(magnitudes));
magnitude_outside_inds = find(M>median_distance_thresh);

median_direction_thresh = 15; %in degrees. Exclude vectors w/ directions > this many degrees from median direction.
A = abs(angles - median(angles));
angle_outside_inds = find(A>median_direction_thresh);

%Exclude vectors that are deviant in EITHER magnitude OR angle:
exclude_inds = union(magnitude_outside_inds,angle_outside_inds);
all_inds = (1:length(angles));
valid_inds = setdiff(all_inds,exclude_inds);

%Update accordingly:
VX = VX(valid_inds);
VY = VY(valid_inds);
X = X(valid_inds);
Y = Y(valid_inds);
NCC= NCC(valid_inds);







%Find the median, get rid of the 5-10% of flow with the largest L1 error:
%M = abs(MAGNITUDES - median(MAGNITUDES));

%Get rid of the flow vectors with the largest magnitude. They often
%represent bad NCC matches of regions that have no counterpart in the
%reference image due to that edge not being in the image or being occluded.
%Could address this by changing NCC threshold parameters, but easier and
%possibly more reliable to just exclude them here:
%[~,magI] = sort(magnitudes,'descend');
%N = length(magnitudes);
%mag_exclude_inds = magI(1:floor(.05*N));
%**You might argue that since there are background and reflection flow
%vectors, and they transform differently, by getting rid of large magnitude
%vectors you might just be getting rid of perfectly good edge flows for
%either the background or reflection, which just happens to move more in
%the scene. However, in practice, since the user is told to move the camera
%only a small amount between images, the absolute magnitudes of both sets
%of flow vectors are relatively small, and also, they are often similar in
%magnitude. In other words, the background pixels only transform slightly
%differently than the reflection pixels, not hugely differently. So
%treating all flow vectors together and getting rid of likely outliers is
%valid.

end