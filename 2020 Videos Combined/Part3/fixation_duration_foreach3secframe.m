datadir='C:\Users\siffa\Desktop\dekstop\trial data\video13';
path = datadir;
participants= dir(strcat(datadir,'\*.mat'));

%Video13/10 tasit score
load('Part3_labeling.mat');
labels= horzcat(num(:,1), num(:,241));

for ch=1:length(participants)
    ROI = zeros(12,7);
    fname=participants(ch).name;
    part_num = str2double(fname(18:22));

    load(strcat(path,'\',fname));
    load(strcat(path,'\',"ROI\",fname));
    
    n = 1;
    
    for i = 1 : 1000 : length(data)
        
        if (i+999) <=length(data)
            datafix = data(i:i+999,[1,14]);
            resultfix = results(i:i+999,:);
        else
            datafix = data(i:length(data),[1,14]);
            resultfix = results(i:length(data),:);
        end
        
        idx = find(datafix(:,2)==1);
        datafix = datafix(idx,:);
        resultfix = resultfix(idx,:);
        
        if isempty(datafix)==1 
            ROI(n,:) = 0;
        elseif isempty(unique(resultfix))==0
            rois = unique(resultfix);
            for j=1:length(rois)
                ROI(n,rois(j)+1) = mean(unique(datafix(find(resultfix(:,1)==rois(j) | resultfix(:,2)==rois(j)),1)));
            end
        end
        n = n +1;
    end
    if ~exist(strcat(path,'\',"Mean_feature_eachROI\"), 'dir')
            mkdir(strcat(path,'\',"Mean_feature_eachROI\"));
    end
    name = strcat(path,'\',"Mean_feature_eachROI\",fname);
    save(name, 'ROI');
    
end

%%barchart for each ROI
datadir='C:\Users\siffa\Desktop\dekstop\trial data\video13';
path = strcat(datadir,'\',"Mean_feature_eachROI\");
participants= dir(strcat(datadir,'\*.mat'));

for row = 1:12
    roihc = [];
    roitb = [];
    
    for ch = 1:length(participants)
    fname=participants(ch).name;
    load(strcat(path,'\',fname));
    participant_id = str2double(fname(18:22));
    label = labels(find(labels(:,1)==participant_id),2);
    
    
%         if round(participant_id/1000)>=12
%             roihc = [roihc;ROI(row,:)];
%         else
%             roitb = [roitb;ROI(row,:)];
%         end
        if label==4
            roihc = [roihc;ROI(row,:)];
        else
            roitb = [roitb;ROI(row,:)];
        end
        
    end
    
x = mean(roihc(:,:),1);
y = mean(roitb(:,:),1);

name = {'Non ROI','EYE','MOUTH','FOREHEAD','NOSE','LEFT CHEEK','RIGHT CHEEK'};
% bar([x', y']), legend('TBI-','TBI+' );
bar([x', y']), legend('Correct','Incorrect' );
% set(gca,'XTick',[1 2 3 4 5])
set(gca,'XTickLabel',name);
set(gca,'XTickLabelRotation',45);
xlabel('Region of Interest (ROI)');
ylabel('Average Fixation Duration in Each ROI');
title(strcat('Bar plot(Video13) for Fixation Duration of Frames'," ", string(round(row*3.33-3.33)), " ",'to'," ",...
    string(round(row*3.33)), ' sec'));
figurepath =".\Figure\Video13\";
saveas(gcf,strcat(figurepath,'Bar plot for Fixation Duration of Frames'," ", string(round(row*3.33-3.33)), " ","to"," ",...
    string(round(row*3.33)), ' sec', '.png'));

close
end



% [row,~] = size(roihc);
% for i= 1:row
%     tb = histogram(roitb(row,:),8,'Normalization','probability');
%     hold on
%     histogram(roihc(row,:),tb.BinEdges,'Normalization','probability');
%     hold off
% end

    
    