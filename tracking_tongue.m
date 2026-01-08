close all
clear variables

% Initialize video reader
% IMPORTANT: Change this path to your video file
videoFile = 'your_video_file.avi';

% Check if video file exists
if ~isfile(videoFile)
    error('Video file not found: %s\nPlease update the videoFile variable with the correct path.', videoFile);
end

vidReader = VideoReader(videoFile);
fprintf('Video loaded: %s\n', videoFile);
fprintf('Duration: %.2f seconds\n', vidReader.Duration);
fprintf('Frame rate: %.2f fps\n', vidReader.FrameRate);

opticFlow = opticalFlowFarneback;

% Different views have different ROIs. Uncomment in order to track the required view.
r = [367.5 350.5 361 365];  % midtest
%r = [135.5 479.5 367 261]; % ltest
%r = [595.5 431.5 412 302]; % rtest

frameRGB = readFrame(vidReader);
frameRGB = imcrop(frameRGB,r);
frameGray = rgb2gray(frameRGB);
pointTracker = vision.PointTracker('MaxBidirectionalError',1);

% Initialize tracking
cp = 1;
rp = 1;
initialize(pointTracker, [cp rp], frameGray);
count = 1;

% Preallocate array for better performance
totalFrames = floor(vidReader.Duration * vidReader.FrameRate);
mpoints = zeros(totalFrames, 2);

fprintf('Processing %d frames...\n', totalFrames);

% Main tracking loop
while hasFrame(vidReader)
    frameRGB1 = readFrame(vidReader);
    frameRGB = imcrop(frameRGB1, r);
    frameGray = rgb2gray(frameRGB);
    frameGray = imsharpen(frameGray);
    flow = estimateFlow(opticFlow, frameGray);
    [row, col] = find(flow.Magnitude == max(flow.Magnitude(:)));
    imshow(frameGray)

    % Threshold for reinitializing tracker
    if max(flow.Magnitude(:)) >= 6
        cp = col;
        rp = row;
        release(pointTracker);
        initialize(pointTracker, [cp rp], frameGray)
    else
        [pt, point_validity] = pointTracker(frameGray);
        cp = pt(1);
        rp = pt(2);
    end

    hold on
    plot(cp, rp, 'ro', 'MarkerSize', 30);
    hold off
    drawnow

    % Store tracked points
    mpoints(count, 1) = cp;
    mpoints(count, 2) = rp;
    count = count + 1;

    % Progress indicator
    if mod(count, 50) == 0
        fprintf('Processed %d frames\n', count);
    end
end

% Trim preallocated array to actual size
mpoints = mpoints(1:count-1, :);
fprintf('Tracking complete. Processed %d frames.\n', count-1);
fprintf('Results stored in variable "mpoints"\n');
