%Modified the read_data.m to create a function to just have the necessary
%variables
clear all;

datadir='..\..\trial data\video9'; %data path in mat format
participants= dir(fullfile(datadir,'*.mat'));
n=length(participants);  % total number of participants

for ch=1:length(participants)
    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22)); %getting paricipant number
  
  load(strcat(datadir,'\',fname));

  name = strcat(datadir,'\',fname);

  X = [diff(data(:,12)),diff(data(:,13))]; %gaze data x,y
  X(end+1,:) = 0;
  data(:,19) = X(:,2) - X(:,1);
  save(name,'data'); %saving the column data in same mat file
  
end


