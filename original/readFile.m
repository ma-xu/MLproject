clear;clc;
data_dir='/home/xuma/data/Anomaly/datasets/';
save_dir='/home/xuma/data/Anomaly/disk/';
filename='20110925';
data= load([data_dir,filename,'.txt']);
data=data(2:end,2:end-1);
%remove the detail failure labels.
data=data(:,1:end-1);

% remove examples contain NaN.
[rows,col]=find(isnan(data));
rows=unique(rows);
data(rows,:)=[];

%split data and label
label=data(:,end);
data=data(:,1:end-1);


%mapminmax will remap lines into a resonable scale.
%data=mapminmax(data');
%data=data';

%Anomaly:-1; Normal:1
% only care about the disk faiure.
anomaly=data(find(label==4),:);
anomaly_label=-ones(size(anomaly,1),1);
normal=data(find(label==0),:);
norm_label=ones(size(normal,1),1);

Struct.data=[anomaly;normal];
Struct.label=[anomaly_label;norm_label];
savepath=[save_dir,filename,'.mat']
save(savepath,'Struct');





