% Step 1: Choose Object Category and Types
object_category = 'Kitchen Utensils';
object_types = {'spoon', 'fork', 'butter_knife', 'cutting_knife', 'ladle'};

% Step 2: Collect Dataset
dataset_dir = '/Users/renugak/ComputerVision/Assignment 3/bow';  % Full path to directory containing subfolders for each object type

image_paths = {};
labels = {};

for i = 1:numel(object_types)
    object_dir = fullfile(dataset_dir, object_types{i});
    images = dir(fullfile(object_dir, '*.jpg'));
    for j = 1:numel(images)
        image_paths = [image_paths; fullfile(object_dir, images(j).name)];
        labels = [labels; object_types{i}];
    end
end

% Create ImageDatastore from image paths
imds = imageDatastore(image_paths);

% Step 3: Preprocess Images
target_size = [100, 100];
preprocessed_images = cell(numel(image_paths), 1);
for i = 1:numel(image_paths)
    try
        % Read image
        image = readimage(imds, i);
        if isempty(image)
            error('Empty image: %s', image_paths{i});
        end
        % Convert to grayscale
        image = rgb2gray(image);
        % Resize image
        image = imresize(image, target_size);
        preprocessed_images{i} = image;
    catch ME
        fprintf('Error reading or preprocessing image file: %s\n', image_paths{i});
        rethrow(ME);
    end
end

% Convert cell array of preprocessed images to numeric array
num_images = numel(preprocessed_images);
image_size = size(preprocessed_images{1});
numeric_images = zeros([image_size, num_images], 'uint8');
for i = 1:num_images
    numeric_images(:,:,i) = preprocessed_images{i};
end

% Step 4: Feature Extraction and Building Visual Vocabulary
% Create bag of features
bag = bagOfFeatures(imds);

% Step 5: Feature Representation
% Encode features using the bag of features
features = encode(bag, numeric_images);

% Step 6: Train Classifier
% Split data into training and testing sets
cv = cvpartition(labels, 'Holdout', 0.2);
train_idx = training(cv);
test_idx = test(cv);

X_train = features(train_idx, :);
y_train = labels(train_idx);
X_test = features(test_idx, :);
y_test = labels(test_idx);

% Train classifier
classifier = fitcecoc(X_train, y_train);

% Step 7: Evaluate Performance
% Predict labels for test set
y_pred = predict(classifier, X_test);

% Evaluate performance
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Display classification report
fprintf('Classification Report:\n');
disp(classification_report(y_test, y_pred));
