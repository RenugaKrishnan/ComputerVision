% Load the template images
templatesFolder = 'objects_to_detect';
templates = cell(1, length(dir(fullfile(templatesFolder, '*.jpg'))) + length(dir(fullfile(templatesFolder, '*.png'))));
index = 1;
for k = 1:length(templates)
    filename = dir(fullfile(templatesFolder, '*.jpg'));
    if isempty(filename)
        filename = dir(fullfile(templatesFolder, '*.png'));
    end
    template = imread(fullfile(templatesFolder, filename(k).name));
    templates{index} = template;
    index = index + 1;
end

% Create the output folder if it doesn't exist
outputFolder = 'detected_objects';
mkdir(outputFolder);

% Initialize list of detected images
detectedImages = {};

% Iterate over each image in the "frames" folder
framesFolder = 'frames';
files = dir(fullfile(framesFolder, '*.jpg'));
files = [files; dir(fullfile(framesFolder, '*.png'))];
for k = 1:length(files)
    filename = files(k).name;
    % Load the scene image
    sceneImage = imread(fullfile(framesFolder, filename));

    % Initialize variables for feature matching
    boxPoints = detectSURFFeatures(rgb2gray(templates{1}));
    scenePoints = detectSURFFeatures(rgb2gray(sceneImage));

    % Extract feature descriptors
    [boxFeatures, boxPoints] = extractFeatures(rgb2gray(templates{1}), boxPoints);
    [sceneFeatures, scenePoints] = extractFeatures(rgb2gray(sceneImage), scenePoints);

    % Match features
    boxPairs = matchFeatures(boxFeatures, sceneFeatures);

    % Handle case where not enough matches are found
    if size(boxPairs, 1) < 4
        continue;
    end

    % Estimate geometric transformation
    try
        tform = estimateGeometricTransform(...
            boxPoints(boxPairs(:, 1)), ...
            scenePoints(boxPairs(:, 2)), ...
            'affine');
    catch ME
        fprintf('Error estimating geometric transformation for %s: %s\n', filename, ME.message);
        continue;
    end

    % Apply transformation to the bounding box
    boxPolygon = [1, 1; size(templates{1}, 2), 1; size(templates{1}, 2), size(templates{1}, 1); 1, size(templates{1}, 1); 1, 1];
    newBoxPolygon = transformPointsForward(tform, boxPolygon);

    % Draw detected object on the scene image
    sceneImage = insertShape(sceneImage, 'Polygon', newBoxPolygon, 'LineWidth', 2, 'Color', 'white');

    % Save the result
    outputFilename = strcat('detected_objects_', filename);
    imwrite(sceneImage, fullfile(outputFolder, outputFilename));
    fprintf('Objects detected in %s and saved at %s\n', filename, fullfile(outputFolder, outputFilename));
    
    % Check if any objects were detected
    detectedImages{end+1} = filename;
end

% Display list of detected images
fprintf('\nList of detected images:\n');
for i = 1:length(detectedImages)
    fprintf('%s\n', detectedImages{i});
end
