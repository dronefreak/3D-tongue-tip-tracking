clearvars -except cameraParams camPoses mpoints lpoints rpoints

% Validate that all required variables exist
if ~exist('cameraParams', 'var')
    error('cameraParams not found. Please run camera calibration first.');
end
if ~exist('camPoses', 'var')
    error('camPoses not found. Please ensure camera poses are defined.');
end
if ~exist('mpoints', 'var')
    error('mpoints not found. Please run tracking_tongue.m first.');
end
if ~exist('lpoints', 'var')
    error('lpoints not found. Please ensure left view tracking is complete.');
end
if ~exist('rpoints', 'var')
    error('rpoints not found. Please ensure right view tracking is complete.');
end

% Validate that all point arrays have the same length
numPoints = length(mpoints);
if length(lpoints) ~= numPoints || length(rpoints) ~= numPoints
    error('Point arrays have different lengths: mpoints=%d, lpoints=%d, rpoints=%d', ...
          numPoints, length(lpoints), length(rpoints));
end

fprintf('Starting 3D reconstruction with %d points\n', numPoints);

figure;
plotCamera(camPoses, 'Size', 0.2);

% Preallocate arrays for better performance
tracks(numPoints) = pointTrack([1 2 3], zeros(3, 2));
xyzPoints = zeros(numPoints, 3);

% Triangulate points from multiple views
for i = 1:numPoints
    points = [lpoints(i, 1), lpoints(i, 2); ...
              mpoints(i, 1), mpoints(i, 2); ...
              rpoints(i, 1), rpoints(i, 2)];
    viewIDs = [1 2 3];
    tracks(i) = pointTrack(viewIDs, points);
    xyzPoints(i, :) = triangulateMultiview(tracks(i), camPoses, cameraParams);

    % Progress indicator
    if mod(i, 100) == 0
        fprintf('Triangulated %d/%d points\n', i, numPoints);
    end
end
grid on

fprintf('Performing bundle adjustment...\n');
% Refine the 3-D world points and camera poses.
[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, cameraParams);

fprintf('Bundle adjustment complete.\n');
fprintf('Mean reprojection error: %.4f pixels\n', mean(reprojectionErrors));

% Display the dense 3-D world points.
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
hold on

% Specify the viewing volume.
loc1 = camPoses.Location{1};
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);

title('Dense Reconstruction');
hold off

fprintf('3D reconstruction complete!\n');
