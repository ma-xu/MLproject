clear;
clc;
addpath ../mcpIncSVM 
addpath ../smote

data_path="/home/xuma/data/Anomaly/disk/data_disk.mat";
load(data_path);

% Rand index data
rng('default');
rand_index=randperm(size(data,1));
data=data(rand_index,:); 
clearvars -except data;

% Split data
batch_size=150;
parts=ceil(size(data,1)/batch_size);
parts_end=(1:parts)'*batch_size;
parts_end(end,1)=size(data,1);

% Initialize
init_size=200;
Init_data=data(1:init_size,:);

mcp_svmtrain(Init_data(:,1:end-1),Init_data(:,end),1,5,0.1);

predict_label=[]; 
for i=init_size+1:size(data,1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    rate=size(predict_normal,1)/size(predict_anomaly,1);
    % generate smote data
    smote_data = smote(predict_anomaly, 3, (rate-1)*100);
    smote_data = [smote_data;predict_anomaly];
    smote_label=-ones(size(smote_data,1),1);
    normal_label=ones(size(predict_normal,1),1);
    train_data=[predict_normal;smote_data];
    train_label=[normal_label;smote_label];
    
    rand_index=randperm(size(train_data,1));
    train_data=train_data(rand_index,:); 
    train_label=train_label(rand_index,:); 
    
    clear rand_index smote_data smote_label rate normal_label;
    
    data_batch=data(batch_size*(i-1)+1:parts_end(i,1),:);
    [predict_label,predict_score]=mcp_svmpredict(data_batch);
    
    
end



