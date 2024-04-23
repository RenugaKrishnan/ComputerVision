% Load the left and right images
I1 = imread("left_ar.jpg");
I2 = imread("right_ar.jpg");

% Check if the images are in RGB format
if ndims(I1) == 3
    % Convert to grayscale using rgb2gray for RGB images
    I1gray = rgb2gray(I1);
else
    % If already grayscale, no conversion needed
    I1gray = I1;
end

if ndims(I2) == 3
    % Convert to grayscale using rgb2gray for RGB images
    I2gray = rgb2gray(I2);
else
    % If already grayscale, no conversion needed
    I2gray = I2;
end

% Display the original images side by side
figure
imshowpair(I1, I2, "montage")
title("I1 (left); I2 (right)")

% Display the composite anaglyph image
figure
imshow(stereoAnaglyph(I1, I2))
title("Composite Image (Red - Left Image, Cyan - Right Image)")

% Detect SURF features in both images
blobs1 = detectSURFFeatures(I1gray, 'MetricThreshold', 2000);
blobs2 = detectSURFFeatures(I2gray, 'MetricThreshold', 2000);

% Display the 30 strongest SURF features in I1
figure
imshow(I1)
hold on
plot(selectStrongest(blobs1, 30))
title("Thirty Strongest SURF Features In I1")

% Display the 30 strongest SURF features in I2
figure
imshow(I2)
hold on
plot(selectStrongest(blobs2, 30))
title("Thirty Strongest SURF Features In I2")

% Extract features from the detected blobs
[features1, validBlobs1] = extractFeatures(I1gray, blobs1);
[features2, validBlobs2] = extractFeatures(I2gray, blobs2);

% Match features between the images
indexPairs = matchFeatures(features1, features2, 'Metric', 'SAD', ...
    'MatchThreshold', 5);

% Select the putatively matched points
matchedPoints1 = validBlobs1(indexPairs(:, 1), :);
matchedPoints2 = validBlobs2(indexPairs(:, 2), :);

% Display the putatively matched points in both images
figure
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2)
legend("Putatively Matched Points In I1", "Putatively Matched Points In I2")

% Estimate the fundamental matrix using RANSAC
[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
    matchedPoints1, matchedPoints2, 'Method', 'RANSAC', ...
    'NumTrials', 10000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);

% Check if enough matching points were found and the epipoles are inside the images
if status ~= 0 || isEpipoleInImage(fMatrix, size(I1)) ...
        || isEpipoleInImage(fMatrix', size(I2))
    error('Not enough matching points were found or the epipoles are inside the images. Inspect and improve the quality of detected features and images.');
end

% Select inlier points
inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

% Display the inlier points in both images
figure
showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2)
legend("Inlier Points In I1", "Inlier Points In I2")

% Estimate stereo rectification
[tform1, tform2] = estimateStereoRectification(fMatrix, ...
    inlierPoints1.Location, inlierPoints2.Location, size(I2));

% Rectify the stereo images
[I1Rect, I2Rect] = rectifyStereoImages(I1, I2, tform1, tform2);

% Display the rectified stereo images
figure
imshow(stereoAnaglyph(I1Rect, I2Rect))
title("Rectified Stereo Images (Red - Left Image, Cyan - Right Image)")
