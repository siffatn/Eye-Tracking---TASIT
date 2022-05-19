clear all;

datadir='C:\Users\siffa\Desktop\trial data\Video9';
path = datadir;
participants= dir(fullfile(datadir,'*.mat'));
threshold = 0.9;
k=1;
for i = 1 : length(participants)    
   fname=participants(i).name;
   part_num = str2double(fname(18:22));
   load(strcat(path,'\',fname));  
   if (sum(isnan(data(:,12)))/length(data))>=threshold     
       participant_drop (k) = part_num;
       k = k+1;
   end
end
  
  
    