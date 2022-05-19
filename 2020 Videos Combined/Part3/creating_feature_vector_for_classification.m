%Load the correlated frames by calling function read_data_{videono}_vectorized_r01
%reading label data (according to question ans)
load('Part3_labeling.mat');
load('tasit_anxiety.mat');
labels= horzcat(num(:,1), num(:,238),num(:,[247,1]));%%%####labels for the population
C1= zeros(1,4);
%categorizing based on no. of correct answers
for i=1:length(labels)    
    if labels(i,2) == 4 
        labels(i,2) = 1;
    else 
        labels(i,2) = 0;
    end
end
% 
name = 'labels for video10 binary.mat';
%save('labels for video13 binary.mat','labels');
load('labels for video10 binary.mat');
%input to directory will change for each video
datadir='..\..\trial data\video10\'; 
kfoldnum = 5;
   
%data is being read for 1000 frames at a time
[X, n, index, participant] = read_data_video10_vectorized_r01(datadir,1,name); % this creates vector X, n gives the sample size
X = fillmissing(X,'constant',0);
% X = medfilt1(X,1); 
% X = X([3,4,9],:); 
% XX = X.^2;
% XXX = X.^3;
% XXXX = X.^4;
% X = vertcat(X,XX,XXX,XXXX);
[~,ii] = ismember(participant,tasit_anxiety(:,1));
tasit_anxiety = tasit_anxiety(ii,2:3);
tasit_anxiety = normalize(tasit_anxiety);
X = vertcat(X,tasit_anxiety');
X = vertcat(X,index',participant');
%csvwrite('Part3 Video10 Vectorized R01.csv', X');

