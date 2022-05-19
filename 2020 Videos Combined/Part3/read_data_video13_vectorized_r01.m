function [X,n, index, participant] = read_data_fn(datadir,~,name)
%%%this function is to get the correlated frames together in one multidimensional array
%%r starting position of data of original time sequence
%%datadir directory of data
%%name passing the labes for each video

path = datadir
participants= dir(strcat(datadir,'*.mat'));
load(name);
n=length(participants);  % total number of participants
X = [];
k=1;
%%appending each of the correlated feature vectors to X
for ch=1:length(participants)
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));
      data1 = data(2000:4999,[17]);  
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
k= 1;
for ch=1:length(participants)  
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));
      data1 = data(9000:9999,[6]);  
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
k= 1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));  
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));  
      data1 = data(10000:10999,[23]);
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
k= 1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));  
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname)); 
      data1 = data(10000:10999,[24]);
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
k= 1;
for ch=1:length(participants)   
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));  
      data1 = data(6000:9000,[18]);
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
k= 1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));  
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));  
      data1 = data(2000:4000,[19]);
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
k= 1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));  
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));  
      data1 = data(4000:5000,[20]);
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
k= 1;
for ch=1:length(participants)    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  [~,rownum]=ismember(part_num, labels(:,1));
  if rownum ~=0
      index(k,1) = labels(rownum,2); 
      participant(k,1) = labels(rownum,4);
      load(strcat(path,'\',fname));  
      data1 = data(2000:4999,[3]);
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
