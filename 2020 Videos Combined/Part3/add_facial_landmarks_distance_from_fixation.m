load('03_10_landmarks_data_ordered.mat');
datadir='..\..\trial data\video10'; %data path in mat format
path = datadir;
participants= dir(fullfile(datadir,'*.mat'));
% n=length(participants);  % total number of participants

nFrames = length(landmarks_ordered);
ROItable = zeros(nFrames,3);

n = 1; %intialization
ROI1=[18 20 25 27 46 48 41 42 37]; %eye
ROI2=[32 34 36 55 11 9 7 49]; %mouth
ROI3=[18 20 25 27 75 72 70]; %forehead
ROI4 = [22 23 43 54 50 40]; %nose
ROI5 = [40 32 50 49 5 3 1 37 42 41]; %left cheek
ROI6 = [43 36 54 55 13 15 17 46 47 48]; %right cheek

for ch=1:length(participants)
    
    n=1;
    fname=participants(ch).name;
    part_num = str2double(fname(18:22));
    
    %%%%%change
    FixP=load(strcat(path,'\',fname));
    FixP.data(isnan(FixP.data))=0;
    %     [row, ~] = find(FixP.data(:,2));
    %
    %
    %     FixP_diff = diff(FixP.data(:,2:3));
    %
    %
    %     [row, ~] = find(FixP_diff);
    
    
    %     if FixP.data(1,2)~=0
    %
    %         row = [0; row];
    %     end
    %
    %     row = row+1;
    %
    %     FixP_fix = [row FixP.data(row,1:3)];
    %
    %     FixP_fix = FixP_fix(FixP_fix(:,3)~=0, 1:4);
    %
    % %     FixP_fix = [row FixP.data(row,1:3)];
    distance1 = zeros(length(FixP.data),4);
    
    for frameNum = 1:length(FixP.data)
        try
            frameString = num2str(floor((frameNum)/12));
            
            xq = FixP.data(frameNum,2); %Load FixP from mat file
            yq = FixP.data(frameNum,3);
            
            if xq>10 && yq>10
                
                xq = round(((xq*800)/1920)-40);
                yq = round(((yq*600)/1080)-13);
                
            end
            
            locs = GetLandmarks(landmarks_ordered, ceil((frameNum)/12), 1);
            %         ROItable(n,1) = ceil(FixP_fix(frameNum,1)/12);
            distance=sqrt((locs(:,1)-xq(1)).^2+(locs(:,2)-yq(1)).^2);
            [distance1(frameNum,1),distance1(frameNum,2)] = min(distance);
            
            
            
            locs = GetLandmarks(landmarks_ordered, ceil((frameNum)/12), 2);
            %         ROItable(n,1) = ceil(FixP_fix(frameNum,1)/12);
            distance=sqrt((locs(:,1)-xq(1)).^2+(locs(:,2)-yq(1)).^2);
            [distance1(frameNum,3),distance1(frameNum,4)] = min(distance);
            
            
            n = n+1;
        catch
            
        end
    end
    
    load(strcat(path,'\',fname));
    data(:,20:23) = distance1;%20,21,22,23
    name = strcat(path,'\',fname);
    save(name, 'data');
    
end




