%Modified the read_data.m to create a function to just have the necessary
%variables
clear all;
load('Part3_labeling.mat'); %all labels for part3
%%change the column number according to the video
labels= horzcat(num(:,1), num(:,238));%%%####labels for the population
C1= zeros(1,4);
feature = [1,2,3,4,5,6,15,16:24,27]; %column number for the features in the data(mat file)

for i=1:length(labels)
    if labels(i,2)== 4 
        labels(i,2) = 1;
    else 
        labels(i,2) = 0;
    end
end

name = 'labels for video10 binary.mat';
% save('labels for video11 binary.mat','labels');
load('labels for video10 binary.mat');

for featureindex = 1:length(feature)
    feat = feature(featureindex);
    for new=1:15   
        try   
            datadir='..\..\trial data\video10'; 
            [X, n, index] = read_data_fn_correlation(datadir,new*1000,name,feat); % this creates vector X, n gives the sample size
            X = fillmissing(X,'constant',0);

            [R,~,P,~] = pointbiserial(index,X',0.05,'np');
            %[corr,p] = corrcoef(X',index); 
            [i,~] = find(P<0.05);

            if isempty(i) ~= 1
                dataframe(featureindex, new) = 1;
            else
                dataframe(featureindex, new) = 0.5;
            end    
        end
    end
end

% save('correlated frames for video13 r01.mat','dataframe');

