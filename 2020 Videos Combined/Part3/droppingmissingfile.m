clear all;

datadir='C:\Users\siffa\Desktop\trial data\Video9';
path = datadir;
participants= dir(fullfile(datadir,'*.mat'));
k=1;
for i = 1 : length(participants)    
   fname=participants(i).name;
   part_num = str2double(fname(18:22));
   load(strcat(path,'\',fname));  
   if (sum(isnan(data(:,12)))/length(data))>=0.9      
       participant_drop (k) = part_num;
       k = k+1;
   end
end
  
  
    