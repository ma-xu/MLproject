clear;
clc;
addpath ../mcpIncSVM 
addpath ../smote
addpath ../original

data_path="/home/xuma/data/Anomaly/disk/data_disk.mat";
load(data_path);

% Rand index data
rng('default');
rand_index=randperm(size(data,1));
data=data(rand_index,:); 
clearvars -except data;

% Split data
batch_size=350;
parts=ceil(size(data,1)/batch_size);
parts_end=(1:parts)'*batch_size;
parts_end(end,1)=size(data,1);

% Initialize
Init_data=data(batch_size*(1-1)+1:parts_end(1,1),:);
predict_anomaly=Init_data(find(Init_data(:,end)==-1),1:end-1);
predict_normal=Init_data(find(Init_data(:,end)==1),1:end-1);
smote_data = smote(predict_anomaly, 3, 10*100);
smote_data = [smote_data;predict_anomaly];
smote_label=-ones(size(smote_data,1),1);
normal_label=ones(size(predict_normal,1),1);
train_data=[predict_normal;smote_data];
train_label=[normal_label;smote_label];
    
rand_index=randperm(size(train_data,1));
train_data=train_data(rand_index,:); 
train_label=train_label(rand_index,:); 



mcp_svmtrain(train_data,train_label,1,5,0.1);

predict_label_all=[];
true_label_all=[];
metricList=[];
for i=2:parts
    
    % generate smote data
    rate=size(predict_normal,1)/size(predict_anomaly,1);
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
    
    % predict
    data_batch=data(batch_size*(i-1)+1:parts_end(i,1),1:end-1);
    [predict_label,predict_score]=mcp_svmpredict(data_batch);
    
    true_label= data(batch_size*(i-1)+1:parts_end(i,1),end);
    metrics=myevaluate(true_label,predict_label');
    metricList=[metricList;metrics];
    
    %add to predict_label_all, true_label_all
    predict_label_all=[predict_label_all;predict_label'];
    true_label_all=[true_label_all;true_label];
    
    % remote data for next train
    predict_anomaly=data_batch(find(data_batch(:,end)==-1),:);
    predict_normal=data_batch(find(data_batch(:,end)==1),:);
    smote_data = smote(predict_anomaly, 3, 10*100);
    smote_data = [smote_data;predict_anomaly];
    smote_label=-ones(size(smote_data,1),1);
    normal_label=ones(size(predict_normal,1),1);
    train_data=[predict_normal;smote_data];
    train_label=[normal_label;smote_label];
    
    rand_index=randperm(size(train_data,1));
    train_data=train_data(rand_index,:); 
    train_label=train_label(rand_index,:);
    
    mcp_svmtrain_next(train_data,train_label,1);
    
end

metric_all=myevaluate(predict_label_all,true_label_all)


