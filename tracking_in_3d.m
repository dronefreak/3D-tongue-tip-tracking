clearvars -except cameraParams camPoses mpoints lpoints rpoints

figure;
plotCamera(camPoses, 'Size', 0.2);

for i=1:length(mpoints)
points = [lpoints(i,1),lpoints(i,2);mpoints(i,1),mpoints(i,2);rpoints(i,1),rpoints(i,2)];
viewIDs = [1 2 3];
tracks(i) = pointTrack(viewIDs,points);
xyzPoints(i,:) = triangulateMultiview(tracks(i), camPoses, cameraParams); 
end
grid on
% Refine the 3-D world points and camera poses.
 [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
         tracks, camPoses, cameraParams);
    
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
