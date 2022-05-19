%Modified the read_data.m to create a function to just have the necessary
%variables
clear all;

datadir='..\..\trial data\video16'; 
participants= dir(fullfile(datadir,'*.mat'));
n=length(participants);  % total number of images

for ch=1:length(participants)
    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));
  
  load(strcat(datadir,'\',fname));

  name = strcat(datadir,'\',fname);
  data(:,17:18) = [data(:,10) - data(:,8),data(:,11) - data(:,9)];
  save(name,'data');
  
end


