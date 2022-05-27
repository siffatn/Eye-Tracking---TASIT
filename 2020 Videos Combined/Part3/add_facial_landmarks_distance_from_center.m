
%run svm_facial_landmarks_distance first
load('03_10_landmarks_data_ordered.mat');
datadir='..\..\trial data\video9'; %data path in mat format
participants= dir(fullfile(datadir,'*.mat'));
path = datadir;

nFrames = length(landmarks_ordered);
ROItable = zeros(nFrames,3);

n = 1; %intialization
ROI1=[18 20 25 27 46 48 41 42 37]; %eye
ROI2=[32 34 36 55 11 9 7 49]; %mouth
ROI3=[18 20 25 27 75 72 70]; %forehead
ROI4 = [22 23 43 54 50 40]; %nose
ROI5 = [40 32 50 49 5 3 1 37 42 41]; %left cheek
ROI6 = [43 36 54 55 13 15 17 46 47 48]; %right cheek
ROI7 = [1:17 27 26 25 24 23 22 21 20 19 18]; %whole face

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
    distance1 = zeros(length(FixP.data),2);
    
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
            %         distance=sqrt((locs(:,1)-xq(1)).^2+(locs(:,2)-yq(1)).^2);
            %         [~,distance1] = min(distance);
            for i = 1:length(ROI7)
                
                index = ROI7(1,i);
                polyface(1,i) = locs(index,1);
                polyface1(1,i) = locs(index,2);
                
            end
            
            polyin = polyshape({polyface},{polyface1});
            [centrx, centry] = centroid(polyin);
            distance = sqrt((centrx-xq(1)).^2+(centry-yq(1)).^2);
            %         angle=acosd((locs(distance1,1)-xq(1))+(locs(distance1,2)-yq(1)))/pi*180;
            [distance1(frameNum,1)] = (distance);
            
            
            
            locs = GetLandmarks(landmarks_ordered, ceil((frameNum)/12), 2);
            for i = 1:length(ROI7)
                
                index = ROI7(1,i);
                polyface(1,i) = locs(index,1);
                polyface1(1,i) = locs(index,2);
                
            end
            
            polyin = polyshape({polyface},{polyface1});
            [centrx, centry] = centroid(polyin);
            distance = sqrt((centrx-xq(1)).^2+(centry-yq(1)).^2);
            %         angle=acosd((locs(distance1,1)-xq(1))+(locs(distance1,2)-yq(1)))/pi*180;
            [distance1(frameNum,2)] = (distance);
            
            
            n = n+1;
        catch
            
        end
    end
    
    load(strcat(path,'\',fname));
    data(:,24:25) = distance1;
    %data = real(data);
    name = strcat(path,'\',fname);
    save(name, 'data');
    
end




