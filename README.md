# Techniques-in-Computer-Vision

The motivation of this project is to put into practice the skills and knowledge acquired in the Computer Vision Module, at the MSc Data Science. The first task of this project will compare the performance of 2 different methods for the extraction of features from images (HOG and SURF), and the classification performance of three models (Random Forest, K-nearest neighbours, and Softmax neural network) for each of the feature extraction methods, resulting in a total of 6 combinations. The second task uses augmented reality in a video to place a 3D object (a red cow) on the palm of a hand.

### Containing files:
#### Computer Vision Report
This pdf file explains the steps needed to build a face recognition model and to perform augmented reality on a given video. There are detail explanations on the use of different feature extraction and classification techniques, and their performance. 
#### RecogniseFace.m
P = RecogniseFace(image,featureType, classifierName) returns a matrix of size Nx3, where N is the number of faces detected in the image. P(:,1) contains their predicted labels. P(:,2) contains the x coordinate of the center point of the face. P(:,3) contains the y coordinate of the center point of the face. Valid arguments for featureType are; ‘HOG’ and ‘SURF’. Valid arguments for classifierName are; ‘KNN’, ‘RF’ and ‘SoftMax’.
#### highFive.m
highFive(videofile) will render a red cow into the palm og the hand for all frames in the video.
Initially, the <highFive> function defines a 3D coordinates system with origin in the center of the hand (x mark).
