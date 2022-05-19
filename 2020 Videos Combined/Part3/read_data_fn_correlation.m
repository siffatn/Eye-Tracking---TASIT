function [X,n, index] = read_data_fn_correlation(datadir,r,name,feat)
%%r starting position of data of original time sequence
%%datadir directory of data
%%name passing the labels for each video

path = datadir;
participants= dir(fullfile(datadir,'*.mat'));
load(name);
n=length(participants);  % total number of participants
k=1;
index = zeros(n,1);

for ch=1:length(participants) 
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);
      img=load(strcat(path,'\',fname));
      img1 = img.data(r:r+999,[feat]); %taking 3s chunks from whole data
      img1 = fillmissing(img1,'constant', 0);
      X(k)=mean(img1);  
      k=k+1;
  end
end

X = X(:,1:k-1);
index = index(1:k-1,1);
% out = find(all(~isnan(X)));
% 
% X = X(:,out);
% index = index(out);
% %normalizing
% X1 = X(1:1000,:);
% % X2 = X(1001:2000, :);
% % X3 = X(2001:3000, :);
% % X4 = X(3001:4000, :);
% % X5 = X(4001:5000, :);
% 
% 
% normX1 = X1 - min(X1(:));
% X1 = normX1 ./ max(normX1(:));
% % normX2 = X2 - min(X2(:));
% % X2 = normX2 ./ max(normX2(:));
% % normX3 = X3 - min(X3(:));
% % X3 = normX3 ./ max(normX3(:));
% % normX4 = X4 - min(X4(:));
% % X4 = normX4 ./ max(normX4(:));
% % normX5 = X5 - min(X5(:));
% % X5 = normX5 ./ max(normX5(:));
% 
X = normalize(X,'range');
% X = vertcat(X1);
% close all;
% 
