close all
clear variables


opticFlow = opticalFlowFarneback;

%Different views have different ROIs. Uncomment in order to track the required view.
r = [367.5 350.5 361 365];  %midtest
%r = [135.5 479.5 367 261]; %ltest
%r= [595.5 431.5 412 302]; %rtest
frameRGB = readFrame(vidReader);
frameRGB = imcrop(frameRGB,r);
frameGray = rgb2gray(frameRGB);
pointTracker = vision.PointTracker('MaxBidirectionalError',1);

cp=1;
rp=1;
initialize(pointTracker,[cp rp],frameGray);
count=1;
while hasFrame(vidReader)
    frameRGB1 = readFrame(vidReader);
    frameRGB = imcrop(frameRGB1,r);
    frameGray = rgb2gray(frameRGB);
    frameGray=imsharpen(frameGray);
    flow = estimateFlow(opticFlow,frameGray);
    [row,col]=find(flow.Magnitude == max(flow.Magnitude(:)));
    imshow(frameGray)
    if max(flow.Magnitude(:))>=6
        cp=col;
        rp=row;
        release(pointTracker);
        initialize(pointTracker,[cp rp],frameGray)
        
    else
     
        [pt,point_validity] = pointTracker(frameGray);
        cp=pt(1);
        rp=pt(2);
    end
    hold on
    plot(cp, rp, 'ro', 'MarkerSize', 30);

    hold off
    drawnow

    prevc=cp;
    prevr=rp;
    mpoints(count,1)=cp;
    mpoints(count,2)=rp;
    count=count+1;
end
