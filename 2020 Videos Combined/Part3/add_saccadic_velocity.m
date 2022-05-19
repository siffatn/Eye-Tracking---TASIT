%Modified the read_data.m to create a function to just have the necessary
%variables
clear all;

datadir='..\..\trial data\video16'; %data path in mat format
participants= dir(datadir);
n=length(participants);  % total number of participants

for ch=1:length(participants)
    
  fname=participants(ch).name;
  part_num = str2double(fname(18:22));%getting paricipant number
  
  load(strcat(datadir,'\',fname));

  name = strcat(datadir,'\',fname);

  %data(:,27) = (data(:,4))./(data(:,1));
  data(:,end+1) = (data(:,4))./(data(:,1));
  
  
  save(name,'data'); %saving the velocity column in the same file
  
end


