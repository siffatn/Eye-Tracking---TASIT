function [X,n, index, participant] = read_data_fn(datadir,~,name)
%%r starting position of data of original time sequence
%%datadir directory of test data
%%name passing the labels for each video
path = datadir;
participants= dir(strcat(datadir,'*.mat'));
load(name);
n=length(participants);  % total number of participants  
X = [];
data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(9000:10000,1); 
      data1(data1>5000) = 0;
      data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(3000:5000,4); 
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
% data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r];  

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(1000:3000,5);  
    %   data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(5000:6000,5);
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)   
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(5000:6000,6);  
      data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(8000:9000,18); 
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(1000:2000,19);  
      %data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(6000:7000,20);  
      %data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(6000:7000,21);  
      % data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(1000:2000,19);  
    %   data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 

data_r = [];
data1 = [];
k=1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2);      
      participant(k,1) = labels(rownum,1);
      load(strcat(path,'\',fname));
      data1 = data(6000:8000,24);  
    %   data1 = filloutliers(data1,'previous','quartiles');
      data1 = fillmissing(data1,'constant', 0);
      data_r(:,k) = data1;
      k = k + 1;
  end  
end
[r,c] = size(data_r);
data_r = reshape(data_r,r*c,1);
data_r = normalize(data_r);
data_r = reshape(data_r,r,c);  
X = [X;data_r]; 
