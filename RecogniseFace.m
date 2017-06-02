% Computer Vision CW 04/2017. Author: Miguel Esteras-Bejar
%% Face Recognition

function P = RecogniseFace(image, featureType, classifierName)
% function returns 'P', containing the people present in the image. 
% 'P' is a matrix of size Nx3, where N is the number of people detected.
% The three columns represent;
% P(:,1) unique label associated with the person on the image
% P(:,2) and f(:,3) is x and y location of the face detected (central point).

% Accepted featureType;     'SURF' -> SURF features
%                           'HOG' -> HOG features

% Accepted classifierName;  'RF' -> Random Forest
%                           'SoftMax' -> Softmax classifier
%                           'KNN' -> K-nearest neighbours

% Pre-process image for face detection

% DCT normalization to achieve illumination invariance
I = uint8(255 * mat2gray(DCT_normalization(rgb2gray(image),40))); 

% detect big faces
faceDetector = vision.CascadeObjectDetector; % create detector object
faceDetector.MergeThreshold = 8;
faceDetector.MinSize = [30 30]; % minimum size square around face
faceDetector.ScaleFactor = 1.1;
boxFaces = step(faceDetector,I);
Faces = cell(1);
if ~isempty(boxFaces)
    for i=1:size(boxFaces,1)
        Faces{i,1} = imresize(imcrop(I,boxFaces(i,:)),[40 40]);  % crop face and normalize image size 
    end
else
    boxFaces = step(faceDetector,I*1.2);
    if ~isempty(boxFaces)
        for i=1:size(boxFaces,1)
            Faces{i,1} = imresize(imcrop(I,boxFaces(i,:)),[40 40]);  % crop face and normalize image size             
        end
    end
end

for i=1:size(boxFaces,1)
    location(1) = boxFaces(i,1)+boxFaces(i,3)/2;
    location(2) = boxFaces(i,2)+boxFaces(i,4)/2;
    Faces{i,2} = location;      % save center point of face
end

Yhat = zeros(size(Faces,1),1); % define empty Yhat

if size(Faces,1)>1
    fprintf('\n%02d faces were detected\n',size(Faces,1))
    
    % Predic label from HOG features
    if strcmp(featureType,'HOG')
        for i=1:size(Faces,1)   % feature extraction
            HOGfeatures(i,:) = extractHOGFeatures(Faces{i,1},'CellSize', [4 4]); 
        end

        if strcmp(classifierName,'KNN')    
            load('KNNhog.mat')
            Yhat = predict(KNNhog,HOGfeatures);

        elseif strcmp(classifierName,'SoftMax')
            load('SMhog.mat')
            pred = SMhog(HOGfeatures');
            [~,idx] = max(pred);
            Yhat = idx'; 

        elseif strcmp(classifierName,'RF')
            load('RFhog.mat');
            pred = predict(cRFhog,HOGfeatures);
            Yhat = [];
            for i=1:size(pred,1)
                Yhat(i,:) = str2double(pred{i});
            end
        end

    elseif strcmp(featureType,'SURF')
        load('surfBag.mat');
        for i=1:size(Faces,1)       % feature extraction
            SURFfeatures(i,:) = single(encode(bag, Faces{i,1}));    
        end

        if strcmp(classifierName,'KNN')
            load('KNNsurf.mat')
            Yhat = predict(KNNsurf,SURFfeatures);

        elseif strcmp(classifierName,'SoftMax')
            load('SMsurf.mat')
            pred = SMsurf(SURFfeatures');
            [~,idx] = max(pred);
            Yhat = idx'; 

        elseif strcmp(classifierName,'RF')
            load('RFsurf.mat');
            pred = predict(RFsurf,SURFfeatures);
            Yhat = [];
            for i=1:size(pred,1)
                Yhat(i,:) = str2double(pred{i});
            end
        end
    end
end

P = zeros(size(Faces,1),3);
P(:,1) = Yhat;                          % labels
P(:,2) = round(boxFaces(:,1)+boxFaces(:,3)/2); % x location
P(:,3) = round(boxFaces(:,2)+boxFaces(:,4)/2); % y location

fprintf('\nAll faces have been identified. Process completed :-)\n')

end
