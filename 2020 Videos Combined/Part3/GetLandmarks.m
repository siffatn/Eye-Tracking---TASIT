%Get a particular face landmarks
%faceNum = 1 for lef-tmost face and 2 for second left-most and so on 

function locs = GetLandmarks(all_landmarks, frameNum, faceNum)

ind = faceNum*2-1;

locs = all_landmarks(:, ind:ind+1,frameNum); 