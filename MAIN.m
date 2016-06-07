% Stanford CS231A Final Project
% Due: 6/6/2016
% Author: G.K.
%% MAIN: 
% Given an ordered series of images taken through a reflective surface (e.g.
% through a window), decompose a reference image into background and 
% reflection layers.

% Dependencies:
% MATLAB Computer Vision Toolbox: the function "EdgesNCC" calls the
% a mex file made assuming that the OpenCV MATLAB plugin is installed in 
% order to use the OpenCV Normalized Cross-Correlation routine.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% SPECIFY ONE OF THESE, OR YOUR OWN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Specify path to directory containing series of images
%Assumes every file within "input_dir" is an image, and that
%consecutive images were taken with slight differences in camera
%orientation.


%Some examples at different image sizes, different background/reflection
%strengths, etc.:
%input_dir = 'INPUT office trees';
%input_dir = 'INPUT indoor library';
%input_dir = 'INPUT vending machine';
%input_dir = 'INPUT Chipotle';
input_dir = 'INPUT picture';
input_dir = 'INPUT vending4';
%For any of the above series that use larger image sizes, you may need to
%extract a subsection of the image later in order to reduce things to a
%reasonable size.

%Image sequences used by other authors, with ground truth layer separations
%(found their data to be "easy" compared to my own collected data, so
%performance on these series is better):
input_dir = 'INPUT rocks';
%input_dir = 'INPUT toys';


%Random seed number for repeatability:
%To NOT seed it, just comment out following line:
random_seed = 12345

%Upper bound on number of pixels for which to get NCC. Set to inf to ignore
NpixUB = 3000%400%inf;

%In RANSAC fitting of 2D perspective transformation, the allowed pixel error
%Unfortunately, the tolerable values are somewhat different for different
%images and have to be set properly or there won't be enough pixels in
%either the background or reflection layers and it won't be able to fit a
%transformation.
%Be strict on error because background and reflection vectors are often close in magnitude & direction, so points from one transformation almost fit the other.
%try < 1. for "rocks"
%try ~ 2. for "toys"
pixelThresh = .1%.1%3.; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%{
if exist('OUTPUT','dir') == 7
    error('OUTPUT folder already exists')
else
    mkdir('OUTPUT')
end
%}

%% Get central reference image
%This is the image whose background and reflection will be separated.

%Get index of reference image:
files_list = dir(fullfile(pwd, input_dir));
files_list = files_list(3:end); %exclude '.' and '..'
Nimages = size(files_list,1);
if mod(Nimages,2) ~= 0
    center_ind = Nimages/2. + .5;
elseif mod(Nimages,2) == 0
    center_ind = Nimages/2.;
end




%% Load central reference image
ref_filename = files_list(center_ind).name;
%im_ref = im2double(imread(fullfile(pwd,'input_dir',ref_filename)));%worked
im_ref = imread(fullfile(pwd,input_dir,ref_filename));


%For testing only: take a subsection of the image:

%Smaller memory reqs
%L = 1;
%R = 150%300;
%T = 1;
%B = 170%200;

%To use the full image (if image is small enough, e.g. "rocks" or "toys"
%series or any of the vending machine series:
L = 1;
R = size(im_ref,2);
T = 1;
B = size(im_ref,1);

%Excerpt image chip:
im_ref = im_ref(T:B,L:R,:);
[H,W,C] = size(im_ref);
Npixels = H*W

%Grayscale reference image to use to get Canny egdes
im_ref_gray = rgb2gray(im_ref);

% Canny Edge Map of reference image
can_ref_2D = edge(im_ref_gray,'canny');
%can_ref_3D = im2double(repmat(can_ref_2D,1,1,3));

%RGB image that is nonzero only at Canny edges
%im_ref_masked = im_ref.*can_ref_3D; %only needed this when previously doing opticalFlow





%% Main loop:
% Iterate through all of the Nimages, separating the motion of background and
% reflection components of combined motion field:

%Container arrays
ALL_COMPOSITE_IMAGES = zeros(H,W,3,Nimages); %The array of input images (composite background + reflection)
ALL_ALIGNED_BG_STACK = zeros(H,W,3,Nimages); %The array of images with backgrounds warped to alignment w/ reference image
ALL_ALIGNED_RFL_STACK = zeros(H,W,3,Nimages); %The array of images with reflections warped to alignment w/ reference image
ALL_EDGES_STACK = zeros(H,W,Nimages);
ALL_BACKGROUND_FLOWS_STACK = zeros(H,W,2,Nimages); %Flow fields such that takes reference image TO other images (so when plotting bg aligned, uses negative of this)
ALL_REFLECTION_FLOWS_STACK = zeros(H,W,2,Nimages); %Flow fields such that takes reference image TO other images (is not visualized, but is used in optimization)
%Don't strictly need warped edges, but good to have:
ALL_EDGES_WARPED_STACK = zeros(H,W,Nimages);

for i=1:Nimages
     
    %Don't compare the reference image to itself:
    if i==center_ind
        %Append the reference image to the stack as is
        %(the flow field is already all 0's)
        ALL_ALIGNED_BG_STACK(:,:,:,i) = im_ref;
        ALL_ALIGNED_RFL_STACK(:,:,:,i) = im_ref;
        ALL_EDGES_STACK(:,:,i) = can_ref_2D;
        ALL_EDGES_WARPED_STACK(:,:,i) = can_ref_2D;
        ALL_COMPOSITE_IMAGES(:,:,:,i) = im_ref;
        
        figure;imshow(im_ref);title('ref');
        figsavename = fullfile(pwd, 'OUTPUT', strcat('Aligned_Warp_',int2str(i),'_ref.png'));
        saveas(gcf,figsavename);  
        close all;
        continue
    end
    
    
    other_filename = files_list(i).name;
    im = imread(fullfile(pwd,input_dir,other_filename));

    %For testing on subsection of image: take chip of image
    im = im(T:B,L:R,:);
    
    %Grayscale reference image to use to get Canny egdes
    im_gray = rgb2gray(im);

    %Canny Edge Map of non-reference image
    can_2D = edge(im_gray,'canny');
    %can_3D = im2double(repmat(can_2D,1,1,3));%not used anymore

    %RGB image that is nonzero only at Canny edges
    %im_masked = im.*can_3D; %only needed this when doing opticalFlow









    %% Normalized Cross-Correlation to find edge flow
    % This is a bottleneck in the code. Can take > few mins per image pair....
    % So could just run once and save the output, then load later on.
    % Switched over to the OpenCV C++ mex code (called within my EdgesNCC function),
    % gives at least an order of magnitude boost in speed.
    NCC_WINDOW_RADIUS = 25 %in pixels %75%100 %Not too important, but ~smaller allows to use pixels at borders of images as well.
    SUBPIX_WINDOW_RADIUS = 1 %in pixels. Assumed odd number. %After peak is located, small window in which to measure centroid to improve location guess %window is (2*SUBPIX_WINDOW_RADIUS + 1) in both dims
    RELATIVE_THRESH = .8; %Decided to not actually use this
    ABSOLUTE_THRESH = 0. %.1 %Set relatively low since using directional threshold, and since doing RANSAC later %Set to 0 to not use at all
    
    [X, Y, VX, VY, NCC] = EdgesNCC(can_ref_2D,can_2D,NCC_WINDOW_RADIUS,SUBPIX_WINDOW_RADIUS,RELATIVE_THRESH,ABSOLUTE_THRESH,NpixUB);

    %For diagnostics
    %[X, Y, VX, VY, NCC] = EdgesNCC(can_ref_2D,can_ref_2D,NCC_WINDOW_RADIUS,SUBPIX_WINDOW_RADIUS,RELATIVE_THRESH,ABSOLUTE_THRESH);%intentional identical
    %[X, Y, VX, VY, NCC] = EdgesNCC(can_ref_2D,circshift(can_ref_2D,20,1),NCC_WINDOW_RADIUS,SUBPIX_WINDOW_RADIUS,RELATIVE_THRESH,ABSOLUTE_THRESH);%intentional identical
    %[X, Y, VX, VY, NCC] = EdgesNCC(can_ref_2D,circshift(circshift(can_ref_2D,20,1),40,2),NCC_WINDOW_RADIUS,SUBPIX_WINDOW_RADIUS,RELATIVE_THRESH,ABSOLUTE_THRESH);%intentional identical
    %[X, Y, VX, VY, NCC] = EdgesNCC(can_2D,can_ref_2D,NCC_WINDOW_RADIUS,SUBPIX_WINDOW_RADIUS,RELATIVE_THRESH,ABSOLUTE_THRESH);%intentionally reversing images

    %Quiver plot of motion vectors
    figure; imshow((can_ref_2D+can_2D)/2.)
    hold on
    quiver(X,Y,VX,VY,0)%0 to keep actual scaling instead of arbitrary MATLAB scaling
    figsavename = fullfile(pwd, 'OUTPUT', strcat('All_Flow_Vectors_',int2str(i),'.png'));
    saveas(gcf,figsavename);
    hold off

    %For additional diagnostics, could look at histogram of magnitudes, angles, NCC vals
    
    
    

    %% RANSAC Homographies
    %Assume that background edge pixels are dominant.
    %Then most edge pixels will transform along with the background, so fit a
    %projective transformation to the edge pixels. Mark inliers as background
    %edge pixels. The remaining pixels are either reflection pixels, or just noise.
    %So fit another projective transformation to the remaining pixels, and mark
    %those inliers as reflection edge pixels.

    %Image1 points
    P1 = vertcat(X',Y',ones(1,length(X)));
    
    %{
    %Diagnostics:
    %To demo that RANSAC pipeline works when have 2 slightly different
    %transformations (w/o noise), uncomment this code
    VX = 50*ones(size(X));
    VX(1:floor(length(X)/2)) = -20;
    VY = 49*ones(size(X));
    VY(1:floor(length(Y)/2)) = -21;
    %}
    
    %Image2 points are the image1 points moved according to the flow field
    P2 = vertcat((X+VX)',(Y+VY)',ones(1,length(X)));
    %P1 and P2 are in format:
    % [[ x1, ..., xN],
    %  [ y1, ..., yN],
    %  [ 1, ...,  1 ]]

    %For repeatability, seed random number generator before doing RANSAC
    if exist('random_seed','var')
        rng(random_seed+5*i+i)
    end
    
    %Get the projective transformation for the background edge pixels
    [bg_inliers, H_bg] = RANSAC_Homography2D(P1,P2,pixelThresh);
    background_pixels = vertcat(X(bg_inliers)',Y(bg_inliers)',VX(bg_inliers)',VY(bg_inliers)');
    background_pixels = double(background_pixels);
    %Make binary mask from indices
    bg_mask = zeros(1,max(bg_inliers(:)));
    for j=1:length(bg_inliers)
        bg_mask(bg_inliers(j)) = 1;
    end

    %Remove background edge pixels
    size(background_pixels);
    size(P1);
    size(bg_mask);
    P1 = P1(:,~bg_mask);
    P2 = P2(:,~bg_mask);
    size(P1);

    %Get the projective transformation for the reflection edge pixels
    [reflection_inliers, H_rfl] = RANSAC_Homography2D(P1,P2,pixelThresh);
    reflection_pixels = vertcat(X(reflection_inliers)',Y(reflection_inliers)',VX(reflection_inliers)',VY(reflection_inliers)');
    reflection_pixels = double(reflection_pixels);

    %Plot the pixels and their flow vectors, this time as 2 separate fields
    figure
    imshow((can_ref_2D+can_2D)/2.)
    hold on
    qb = quiver(background_pixels(1,:),background_pixels(2,:),background_pixels(3,:),background_pixels(4,:),0);
    qb.Color = 'blue';
    qr = quiver(reflection_pixels(1,:),reflection_pixels(2,:),reflection_pixels(3,:),reflection_pixels(4,:),0);
    qr.Color = 'red';
    figsavename = fullfile(pwd, 'OUTPUT', strcat('Separated_Flow_',int2str(i),'.png'));
    saveas(gcf,figsavename);
    
    %An alternative to the greedy RANSAC used above is the joint RANSAC
    %approach discussed in Section 3.3 of my paper.




    %% Interpolate the sparse motion field to get a dense motion field

    'Interpolating sparse motion field to get dense motion field...'
    
    % A few options:
    % 1: Use very many of the pixels, then do better cubic interp with interp2 or griddata. 
    % But then get nans outside the convex hull of points ued to fit function
    % 2: Use scatteredInterpolant class which gives flow over whole image.
    % But only supports nearest and linear, not cubic interpolation.
    % 3. Use cubic interpolation inside since presumably better, then use
    % linear interpolation for points outside convex hull.
    %
    % To be consistent with warping matrix used later [Section 3.4], use bilinear interpolation
    %
    % It's good to get flow vectors from all over image, including border,
    % since then the motion field will be more accurate.

    %Sample point coordinates
    [xi, yi] = meshgrid((1:size(can_ref_2D,2)),(1:size(can_ref_2D,1)));
    
    %Use the scatteredInterpolant class: linear interpolation
    bg_VX_dense = scatteredInterpolant(background_pixels(1,:)',background_pixels(2,:)',background_pixels(3,:)');
    bg_VX_dense = bg_VX_dense(xi,yi);

    bg_VY_dense = scatteredInterpolant(background_pixels(1,:)',background_pixels(2,:)',background_pixels(4,:)');
    bg_VY_dense = bg_VY_dense(xi,yi);

    rfl_VX_dense = scatteredInterpolant(reflection_pixels(1,:)',reflection_pixels(2,:)',reflection_pixels(3,:)');
    rfl_VX_dense = rfl_VX_dense(xi,yi);

    rfl_VY_dense = scatteredInterpolant(reflection_pixels(1,:)',reflection_pixels(2,:)',reflection_pixels(4,:)');
    rfl_VY_dense = rfl_VY_dense(xi,yi);
    
    
    %{
    %Using biharmonic spline interpolation is MUCH slower, but gives
    smoother motion field, which may be helpful in improving initial motion
    field estimates.
    bg_VX_dense = griddata(background_pixels(1,:),background_pixels(2,:),background_pixels(3,:),xi,yi,'v4');
    bg_VY_dense = griddata(background_pixels(1,:),background_pixels(2,:),background_pixels(4,:),xi,yi,'v4');
    rfl_VX_dense = griddata(reflection_pixels(1,:),reflection_pixels(2,:),reflection_pixels(3,:),xi,yi,'v4');
    rfl_VY_dense = griddata(reflection_pixels(1,:),reflection_pixels(2,:),reflection_pixels(4,:),xi,yi,'v4');
    %}

    
    %Background layer
    bgflow(:,:,1) = bg_VX_dense;
    bgflow(:,:,2) = bg_VY_dense;
    bgimflow = flowToColor(bgflow);
    figure;imshow(bgimflow);  
    figsavename = fullfile(pwd, 'OUTPUT', strcat('Background_Flow_',int2str(i),'.png'));
    saveas(gcf,figsavename);    
    
    %Reflection layer
    rflflow(:,:,1) = rfl_VX_dense;
    rflflow(:,:,2) = rfl_VY_dense;
    rflimflow = flowToColor(rflflow);
    figure;imshow(rflimflow);
    figsavename = fullfile(pwd, 'OUTPUT', strcat('Reflection_Flow_',int2str(i),'.png'));
    saveas(gcf,figsavename);  






    %% Warping images: warp TO the reference image [TO direction used for alignment vs. FROM used for later optimization]
    
    %Warping according to dense BACKGROUND motion field to align with reference image
    im_warped_to_bg = WARP(im_ref,im,-bgflow(:,:,1),-bgflow(:,:,2));
       
    figure;imshow(im_warped_to_bg);title(strcat('Warped ', int2str(i)));
    figsavename = fullfile(pwd, 'OUTPUT', strcat('Aligned_Warp_',int2str(i),'.png'));
    saveas(gcf,figsavename);  
    
    %figure;imshow(im);title('im');
    %figure;imshow((im_warped+im_ref)/2.);title('ref + warped');
    
    
    %{
    %DIAGNOSTICS: switch reference image, negate motion field. This warps
    the reference image to the other images
    im_warped_to_bg__reverse = WARP(im,im_ref,bg_VX_dense,bg_VY_dense);
    figure;imshow(im_warped_to_bg__reverse);title(strcat('Warped ', int2str(i)));
    figsavename = fullfile(pwd, 'OUTPUT', strcat('Aligned_Warp_',int2str(i),'_reversed.png'));
    saveas(gcf,figsavename);  
    %}
    
    



    %Warping according to dense REFLECTION motion field to align with reference image
    %(this one is not visualized in this code, but is required for the
    %optimization step later on)
    im_warped_to_rfl = WARP(im_ref,im,-rflflow(:,:,1),-rflflow(:,:,2));
    
    
    %Append arrays to the container stacks
    ALL_ALIGNED_BG_STACK(:,:,:,i) = im_warped_to_bg;
    ALL_ALIGNED_RFL_STACK(:,:,:,i) = im_warped_to_rfl;
    ALL_EDGES_STACK(:,:,i) = can_2D;
    ALL_BACKGROUND_FLOWS_STACK(:,:,:,i) = bgflow;
    ALL_REFLECTION_FLOWS_STACK(:,:,:,i) = rflflow;
    ALL_COMPOSITE_IMAGES(:,:,:,i) = im;
    
    
    %Also, align the edge images using the same background motion field:
    [ALL_EDGES_WARPED_STACK(:,:,i), ~] = warpFL(double(can_2D),-bgflow(:,:,1),-bgflow(:,:,2));


    %Close all the figures since they are saved anyway
    close all;
end
%%%%%%%%%%%%
%end of main loop over images
%%%%%%%%%%%%








%% Getting initial layer decomposition [Section 3.3]

% Since all images are warped to alignment with background, background 
% edges are stationary and reflection edges move across images. So taking
% the mean/median can give an idea of which pixels belong to background
% edges vs. reflection edges, since background edges add coherently while
% reflection edges add incoherently. The following are a few ways of trying
% to separate the edge and background edge pixels by taking advantage of
% motion differences: mean, median, multiplication, multiplication after
% Gaussian KDE:

% Depthwise MEAN over stack of images
edge_stack_mean = mean(ALL_EDGES_WARPED_STACK,3);
figure;imshow(edge_stack_mean);title('edge_stack_mean');
saveas(gcf, fullfile(pwd, 'OUTPUT', 'edge_stack_mean.png'));

% Depthwise MEDIAN over stack of images
% Gives surprisingly good separation of background/reflection edges.
edge_stack_median = median(ALL_EDGES_WARPED_STACK,3);
figure;imshow(edge_stack_median);title('edge_stack_median');
saveas(gcf, fullfile(pwd, 'OUTPUT', 'edge_stack_median.png'));

%Depthwise pixel multiplication
coherent_stack = prod(ALL_EDGES_WARPED_STACK,3);
coherent_stack = mat2gray(coherent_stack);
figure;imshow(coherent_stack);title('coherent_stack');
saveas(gcf, fullfile(pwd, 'OUTPUT', 'coherent_edges.png'));

% The above depthwise multiplication by binary masks gives a lot of noise
% since even for a true edge, if there is a single image in the stack that
% is not an edge at that pixel, that edge pixel is lost by multiplication.
% So, treat each pixel as a probability of being an edge. There is some
% uncertainty in the exact location of the edge. Just say the uncertainty
% is normally distributed with FWHM ~1 pixel. Stationary background edges
% will be built up by elementwise multiplication, while nonstationary
% reflection edges will be damped down by multiplication:
%Gaussian KDE:
for jj=1:Nimages
    %Default SD=.5, corresponding to FWHM=1, so leave at default setting:
    edge_stack_KDE(:,:,jj) = imgaussfilt(ALL_EDGES_WARPED_STACK(:,:,jj));
end
%Depthwise pixel multiplicationon KDE stack
coherent_stack_KDE = prod(edge_stack_KDE,3);
coherent_stack_KDE = coherent_stack_KDE/nansum(coherent_stack_KDE(:)); %Normalize
coherent_stack_KDE = mat2gray(coherent_stack_KDE);
figure;imshow(coherent_stack_KDE);title('coherent_stack_KDE');
saveas(gcf, fullfile(pwd, 'OUTPUT', 'coherent_edges_KDE.png'));

%Make edge map of the reflection:
edge_reflection = max(can_ref_2D-edge_stack_median,0);%clipping low end at 0
figure;imshow(edge_reflection);title('edge_reflection');
saveas(gcf, fullfile(pwd, 'OUTPUT', 'edge_reflection.png'));





%% Separate initial background and reflection images [Section 3.3]:
background = zeros(size(im_ref));
%bg_size = size(background);




% MINIMUM estimate
[channelwise_bg_min,~] = min(ALL_ALIGNED_BG_STACK,[],4);
%channelwise_bg_min = channelwise_bg_min/255.;
figure;imshow(channelwise_bg_min);
saveas(gcf,fullfile(pwd, 'OUTPUT', 'Estimated_Background_Initial_Min.png'));

%[channelwise_max,~] = max(ALL_ALIGNED_IMAGES_STACK,[],4);
%figure;imshow(channelwise_max);
%saveas(gcf, fullfile(pwd, 'OUTPUT', 'Initial_Max.png'));

% MEDIAN estimate
channelwise_bg_median = median(ALL_ALIGNED_BG_STACK,4);
%channelwise_bg_median = channelwise_bg_median/255.;
figure;imshow(channelwise_bg_median);
saveas(gcf, fullfile(pwd, 'OUTPUT', 'Estimated_Background_Initial_Median.png'));



%To do C++ NCC had to convert to uint8. Now convert back:
im_ref = double(im_ref)/255.;




%If using the MINIMUM as the background estimate (as in Xue et al.)
channelwise_rfl_min = im_ref-channelwise_bg_min;
figure;imshow(channelwise_rfl_min);
figsavename = fullfile(pwd, 'OUTPUT', 'Estimated_Reflection_Initial_Min.png');
saveas(gcf,figsavename); 

%To make more easily visible, enhance contrast to see backgroudn better:
reflection = imadjust(channelwise_rfl_min,[.0,.1],[]);
figure; imshow(reflection);
figsavename = fullfile(pwd, 'OUTPUT', 'Estimated_Reflection_Initial_Imadjust_Min.png');
saveas(gcf,figsavename); 




%If using the MEDIAN as the background estimate (for comparison)
channelwise_rfl_median = im_ref-channelwise_bg_median;
figure;imshow(channelwise_rfl_median);
figsavename = fullfile(pwd, 'OUTPUT', 'Estimated_Reflection_Initial_Median.png');
saveas(gcf,figsavename); 

%To make more easily visible, enhance contrast to see background better:
reflection = imadjust(channelwise_rfl_median,[0., .1],[]);
figure; imshow(reflection);
figsavename = fullfile(pwd, 'OUTPUT', 'Estimated_Reflection_Initial_Imadjust_Median.png');
saveas(gcf,figsavename); 

%Close all since save anyway
close all;













%% Optimization

% Set up initial values:


%Solution vector, b
%b vector is the flattened vector of all color channels of all input images
b = reshape(ALL_COMPOSITE_IMAGES,numel(ALL_COMPOSITE_IMAGES),1);
b = b/255.; %to scale to [0,1]


%Initial guess for alpha matte (assume it's constant across image)
% 0 <= alpha <= 1
alpha_hat = .5;
beta_hat = 1. - alpha_hat;
% .5 is arbitrary, used since in middle of [0,1]. Possibly could improve
% starting guess by simple rules of thumb looking at image intensities or
% simple measurements of scene/environment physics. But possibly not too
% important since if doing iterative optmization can just go back and solve
% for alpha again after updating values of other variables.


% Initial estimates of background and reflection layers:
%Decide if using the min or median version. Xue et al. used min, so do
%that. Either way, must be consistent and use same version for reflection
%and background layers.
IB_hat = channelwise_bg_min; %channelwise_bg_median
IR_hat = channelwise_rfl_min; %channelwise_rfl_median

%Reshaping for convenience later
IB_hat = reshape(IB_hat,numel(IB_hat),1);
IR_hat = reshape(IR_hat,numel(IR_hat),1);





% OVERVIEW:

%First, hold the motion fields constant and improve the estimate of the
%background and reflection layers for the reference image, as well as
%improving the alpha constant value.


%[Future additional capability]: do as in Xue et al.:
%Can use the improved image estimates and alpha constant value, go back
%and solve for the motion fields.
%Alternate between these 2 optimization steps, iteratively refining the
%decomposition into background/reflection, and the motion fields of those 2
%separated images.
nIters = 1; %If in future wanted to do the alternating method, set this to 2 or higher
for iter=1:nIters
    
    %% Part 1: 
    % Hold motion fields constant and solve for background/reflection images
    
    
    % Get the warping matrices:
    % For both the background and reflection layers, and for all 3 color channels,
    % get the warping matrix that transforms the REFERENCE image to the index i
    % image. The warping matrix is the same for all color channels of a given
    % image index i.
    % Each warping matrix is Npixels x Npixels, but is sparse.
    
    % Build the A matrix to form the equation Ax=b
    % See Section 3.4 for how I derived the A matrix.

    % Since MATLAB sparse arrays cannot be > 2 dimensional, cannot store
    % various W matrices conveniently in a container array, so just directly
    % build the constraint matrix from the various W and other terms:

    % Create the linearized system of equations [following my Section 3.4]:
    A = sparse([]);
    for i=1:Nimages
        
        %For the reference image, the warping matrix is just identity matrix:
        if i == center_ind
            vec = IB_hat - IR_hat;
            chunk = horzcat(beta_hat*speye(3*Npixels),alpha_hat*speye(3*Npixels),vec);
            A = vertcat(A,chunk);

        else
            %Get background and reflection warping matrices for index i image
            W_B = GetWarpMatrix(ALL_BACKGROUND_FLOWS_STACK(:,:,:,i)); %Use +, not multiply by -1 since direction is already REF -> other
            W_R = GetWarpMatrix(ALL_REFLECTION_FLOWS_STACK(:,:,:,i)); %Use +, not multiply by -1 since direction is already REF -> other
            %sizeWB = size(W_B)
            %sizeWR = size(W_R)

            R_temp = horzcat(beta_hat*W_R,sparse(Npixels,2*Npixels));
            R_block = vertcat(R_temp,circshift(R_temp,Npixels,2),circshift(R_temp,-Npixels,2));

            B_temp = horzcat(alpha_hat*W_B,sparse(Npixels,2*Npixels));
            B_block = vertcat(B_temp,circshift(B_temp,Npixels,2),circshift(B_temp,-Npixels,2));
            
            top = W_B*IB_hat(1:Npixels) - W_R*IR_hat(1:Npixels);
            mid = W_B*IB_hat(Npixels+1:2*Npixels) - W_R*IR_hat(Npixels+1:2*Npixels);
            bottom = W_B*IB_hat(2*Npixels+1:end) - W_R*IR_hat(2*Npixels+1:end);
            vec = vertcat(top,mid,bottom);
            %sizevec = size(vec)

            chunk = horzcat(R_block,B_block,vec);
            A = vertcat(A,chunk); 
        end
    end
    
    
    %Initial estimate of variables
    x0 = vertcat(IR_hat,IB_hat,alpha_hat);
    
    %Make everything sparse to feed into optimization
    A = sparse(A);
    b = sparse(b);
    x0 = sparse(x0);
    A = sparse(A);
    
    
    
    
    %d=NOTdoingOPTIM

    %Try to solve the system of equations using ordinary least squares, just to see what it gives:
    %These approaches do not work due to memory contstraints. 
    %Also, they cannot take advantage of the initial solution.
    %sol = sparse(A)\b;
    %sol = pinv(sparse(A))*b;
    %[U,S,V] = svds(A);
    
    
    % Using MATLAB constrained linear least squares: "lsqlin"
    % Not surprisingly, gives bad result since doesn't take advantage of
    % initial solution and uses L2 norm which is not robust.
    %{
    %Constraint on image intensities and alpha: must be on [0,1]:
    lb = sparse(length(x0),1);
    ub = ones(length(x0),1);
    %Set optimization options
    %options = optimoptions('lsqlin','Algorithm','active-set','MaxIterations',5,'Display','iter')
    options = optimoptions('lsqlin','Algorithm','trust-region-reflective','MaxIterations',200,'Display','iter')
    %Solve the problem
    'Solving system of equations for Ireflection, Ibackground, alpha...'
    sol = lsqlin(A,b,[],[],[],[],lb,ub,x0,options);
    %}
    
    
    % Solve using custom IRLS implementation using penalty method to enforce upper/lower bounds:
    
    nIter=20; %20 iterations of IRLS
    p=1; %Use the L1 norm %Could consider implementing approach to gradually change it from p=2 at start
    sol = IRLS(A,b,p,x0,nIter);
    
    
    
    
    % Update the variables we just solved for:
    IR_hat = sol(1:(3*Npixels));
    IB_hat = sol((3*Npixels)+1:end-1);
    alpha_hat = sol(end);
    
    
    %% Part 2: Could implement this later for added improvement:
    % Like in Xue et al., solve for the motion fields by keeping everything else fixed
    %'Solving system of equations for reflection and background motion fields...'
    %...

    
    
end
    
    
    
    
    
    
    
    

    
    
%% Reshape the final output to be in the right format

%Reshape the various variables
%IR_final = reshape(sol(1:(3*Npixels)),H,W,3);
%IB_final = reshape(sol((3*Npixels)+1:end-1),H,W,3);
%alpha_final = sol(end);
IR_final = reshape(IR_hat,H,W,3);
IB_final = reshape(IB_hat,H,W,3);
alpha_final = alpha_hat;




%View final results
figure; imshow(IR_final),title('Final Reflection')
figure; imshow(IB_final),title('Final Background')


%Basic Diagnostics
%{
format long
max(IR_final(:))
max(IB_final(:))
median(IB_final(:))
median(IR_final(:))
%}


%Comparison to ground truth:
%If also had a ground truth decomposition, could compare the similarity of
%the decomposed estimates vs. the ground truth decomposition, e.g. using
%NCC or mutual information as a siilarity metric. In practice, just visual
%inspection is decent way of juding effectiveness of algorithm.

%WHAT NEXT?
%Read Section 5 (Conclusion) of my paper to hear discussion of future
%improvements. There are a lot of interesting things to explore with this
%problem, much more than can fit into a single quarter, especially
%considering the complexity of this topic for a team of 1 person.