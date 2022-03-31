load('03_10_landmarks_data_ordered_with_forehead.mat');
datadir='C:\Users\siffa\Desktop\dekstop\trial data\video10';
path = datadir;
participants= dir(strcat(datadir,'\*.mat'));

nFrames = length(landmarks_ordered);
ROItable = zeros(nFrames,3);

n = 1; %intialization
ROI1=[18 19 20 21 22 23 24 25 26 27 46 47 48 43 28 40 41 42 37]; %eye
ROI2=[32 34 36 55 11 9 7 49]; %mouth
ROI3=[18 20 25 27 75 72 70]; %forehead
ROI4 = [28 32 33 34 35 36]; %nose
ROI5 = [40 32 50 49 5 3 1 37 42 41]; %left cheek
ROI6 = [43 36 54 55 13 15 17 46 47 48]; %right cheek

for ch=1:length(participants)
    
    n=1;
    fname=participants(ch).name;
    part_num = str2double(fname(18:22));

    %%%%%change
    FixP=load(strcat(path,'\',fname));
    FixP.data(isnan(FixP.data))=0;

    results = zeros(length(FixP.data),1);
    for frameNum = 1:length(FixP.data)
    try
%         frameString = num2str(floor((frameNum)/12));
        
        xq = FixP.data(frameNum,2); %Load FixP from mat file
        yq = FixP.data(frameNum,3);

        if xq>10 && yq>10

        xq = round(((xq*800)/1920)-40);
        yq = round(((yq*600)/1080)-13);

        end

        locs = GetLandmarks(landmarks_ordered, ceil((frameNum)/12), 1);        
        
        if xq>0 || yq>0
            if inpolygon(yq,xq,locs(ROI1,1),locs(ROI2,2))==1
               results(frameNum,1) = 1;
            elseif inpolygon(yq,xq,locs(ROI2,1),locs(ROI2,2))==1
               results(frameNum,1) = 2;
            elseif inpolygon(yq,xq,locs(ROI3,1),locs(ROI3,2))==1
               results(frameNum,1) = 3;
            elseif inpolygon(yq,xq,locs(ROI4,1),locs(ROI4,2))==1
               results(frameNum,1) = 4;
            elseif inpolygon(yq,xq,locs(ROI5,1),locs(ROI5,2))>=1
               results(frameNum,1) = 5;
            elseif inpolygon(yq,xq,locs(ROI6,1),locs(ROI6,2))==1
               results(frameNum,1) = 6;            
            else 
               results(frameNum,1) = 0;
           end
        end
        
        %face2
        locs = GetLandmarks(landmarks_ordered, ceil((frameNum)/12), 2);
        if xq>0 || yq>0
            if inpolygon(yq,xq,locs(ROI1,1),locs(ROI2,1))==1
               results(frameNum,2) = 1;
            elseif inpolygon(yq,xq,locs(ROI2,1),locs(ROI2,2))==1
               results(frameNum,2) = 2;
            elseif inpolygon(yq,xq,locs(ROI3,1),locs(ROI3,2))==1
               results(frameNum,2) = 3;
            elseif inpolygon(yq,xq,locs(ROI4,1),locs(ROI4,2))==1
               results(frameNum,2) = 4;
            elseif inpolygon(yq,xq,locs(ROI5,1),locs(ROI5,2))>=1
               results(frameNum,2) = 5;
            elseif inpolygon(yq,xq,locs(ROI6,1),locs(ROI6,2))==1
               results(frameNum,2) = 6;
            else 
               results(frameNum,2) = 0;
            end
        end
        n = n+1;
    catch

    end
    end
    name = strcat(path,'\',"ROI\",fname);
    save(name, 'results'); 
%     
end




