% 04/2017 Miguel Esteras-Bejar
%% Tracking and augmented reality 
function highFive(filename)

HandVideo = VideoReader(filename);

% Describe initial 2D coordenates of fingertips
I = readFrame(HandVideo);             % Load image
load cameraParams;                    % Load cameraParams
J = undistortImage(I, cameraParams);  % Undistort image

q0=[434 287]; q1=[571 255]; q2=[486 143]; q3=[438 135]; q4=[394 154]; q5=[345 195]; 
initial_q = vertcat(q0,q1,q2,q3,q4,q5);

% Define p points in 3D world in mm (10mm = 32pixels)
p0=[0 0]; p1=[42.8 -10]; p2=[16.3 -45]; p3=[1.2 -47.5]; p4=[-12.5 -41.6]; p5=[-27.8 -28.8];
p_points = vertcat(p0,p1,p2,p3,p4,p5);

% 3D points matrix
P = zeros(4,6);
for i=1:6
    P(1, i) = p_points(i,1);    % x 
    P(2, i) = p_points(i,2);    % y
    P(3, i) = 0;                % z
    P(4, i) = 1;                % homogenised
end

%% Camare parameters

% camera intrinsic matrix
K = cameraParams.IntrinsicMatrix';

% camera extrinsics
x = [0, 0, 0, 0, 0, 500];

%% Track points in video

HandVideo = VideoReader(filename);
pointTracker = vision.PointTracker();
initialize(pointTracker, initial_q, I);
NoFrames = HandVideo.Duration*HandVideo.FrameRate;
q = cell(1,NoFrames+1); q{1} = initial_q;   % record q points for every frame
M = cell(1,NoFrames);                       % record projectionmatrix M for every frame

for f = 1:NoFrames
    
    I = readFrame(HandVideo);
    I = undistortImage(I, cameraParams);    % Undistort image
    % Track the points. 
    [pts, isFound] = step(pointTracker, I);
    q{f+1} = q{f};                          % in case some points are lost
    q{f+1}(isFound,:) = pts(isFound, :);    % update q points with tracker
    setPoints(pointTracker, q{f+1});        % update tracker
    
    % extimate projection errors and update x
    x_estimate = lsqnonlin(@(x)ReprojectionError(x, K, P, q{f+1}'), x) ;
    x = x_estimate;
    
    % rotation elements
    thetax = x_estimate(1); thetay = x_estimate(2); thetaz = x_estimate(3);

    % T; [tx ty tz]'
    tx = x_estimate(4); ty = x_estimate(5); tz = x_estimate(6);

    % R; [Rx Ry Rz]'
    Rx = [1 0 0; 0 cos(thetax) -sin(thetax); 0 sin(thetax) cos(thetax)];
    Ry = [cos(thetay) 0 sin(thetay); 0 1 0; -sin(thetay) 0 cos(thetay)];
    Rz= [cos(thetaz) -sin(thetaz) 0; sin(thetaz) cos(thetaz) 0; 0 0 1];

    % Rotation matrix R
    R = Rz * Ry * Rx;

    % Translation vectors T 
    T = [tx ty tz]';

    % Projection matrix M
    M{f} = K*[R, T];
    
end
release(pointTracker);

%% Project 3D object

% load 3D object
[V,F] = LoadOBJFile('cow4.obj');

HandVideo = VideoReader(filename);
for f = 1:NoFrames
    I = readFrame(HandVideo);
    image(I); hold on;
    Project3DSurface(V, F, M{f}, [0, 0, 0], 5, [0.9,0,0]); hold off;   
end

end