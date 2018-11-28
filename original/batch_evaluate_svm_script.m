%{
    Description:    Evaluate whole disk data without feature selection and plotting.
    Author:         Xu Ma.
    Date:           Nov/14/2018
    Email:          xuma@my.unt.edu
%}

%% import data
load('/home/xuma/data/Anomaly/datasets/data.mat');

%% Rand index data
rng('default');
rand_index=randperm(size(data,1));
data=data(rand_index,:); 

%% Split data
batch_size=150;
parts=ceil(size(data,1)/batch_size);
parts_end=(1:parts)'*batch_size;
parts_end(end,1)=size(data,1);

%% init predict bounder support vectors.
predict_normal=[];
predict_anomaly=[];
Sensitivity=[];
Specifcity=[];
F1=[];
metricList=[];

%%Initialize
Init_data=data(batch_size*(1-1)+1:parts_end(1,1),:);
Init_anomaly=Init_data(find(Init_data(:,end)==-1),1:end-1);
Init_normal=Init_data(find(Init_data(:,end)==1),1:end-1);
predict_anomaly=[predict_anomaly;Init_anomaly];
predict_normal=[predict_normal;Init_normal];




%% For loop.
for i=2:parts
    
    %% Training model using predict_bsv and predict_anomaly
    rate=size(predict_normal,1)/size(predict_anomaly,1);
    if(rate>1)
        smote_data = smote(predict_anomaly, 3, (rate-1)*100);
        smote_data=[smote_data;predict_anomaly];
    else
        smote_data = [predict_anomaly];
    end
    bsv_label=ones(size(predict_normal,1),1);
    smote_label=-ones(size(smote_data,1),1);
    train_data=[predict_normal;smote_data];
    train_label=[bsv_label;smote_label];
    rand_index=randperm(size(train_data,1));
    train_data=train_data(rand_index,:); 
    train_label=train_label(rand_index,:); 

    [Parameters]=validate_svm([train_data train_label]);
    %% Parameters for kernel SVM
    c=num2str(Parameters.c); % The cofficient of slack varbles.
    t=num2str(Parameters.t); % kerfunction. 0linear; 1Polynomial; 2RBF; 3Sigmoid
    g=num2str(Parameters.g);%gamma value for poly/rbf/sigmoid kernel.
    
    train_command=['-q',' -t ',t,' -g ',g, ' -c ',c];
    %train_command='-s 5 -t 2 -g 0.1 -n 0.01 -c 0.1';
    model=libsvmtrain(train_label,train_data,train_command);
    %     
%     %% Get bounder support vectors
%     bsv_index_2=find(model.sv_coef<c);
%     predict_bsv=full(model.SVs(bsv_index_2,:));
%     predict_bsv=[predict_bsv ones(size(predict_bsv,1),1)];

    %% Handle data
    data_batch=data(batch_size*(i-1)+1:parts_end(i,1),:);
    
    %% Train model
    %pause(3);
    [predict_label,accuracy,score]=libsvmpredict(data_batch(:,end),data_batch(:,1:end-1),model,'-q');
    
    % get metrics 
    metrics=myevaluate(data_batch(:,end),predict_label);
    metrics.sensitivity 
    metrics.specificity
    metrics.F1_measure
    Sensitivity=[Sensitivity;metrics.sensitivity];
    Specifcity=[Specifcity;metrics.specificity];
    F1=[F1;metrics.F1_measure];
    metricList=[metricList;metrics];
    
    %% Add top 30% anomaly and top 30% normal.
    [~,index]=sort(score,'descend');
    anomaly_index=index(end-floor(length(find(score<0))*0.3):end);
    normal_index=index(1:floor(length(find(score>0))*0.3));
    anomaly_add=data_batch(anomaly_index,1:end-1);
    normal_add=data_batch(normal_index,1:end-1);
    
    predict_normal=[predict_normal;normal_add];
    predict_anomaly=[predict_anomaly;anomaly_add];
    
end

%% remove examples contain NaN.
[rows,col]=find(isnan(Sensitivity));
rows=unique(rows);
Sensitivity(rows,:)=[];
[rows,col]=find(isnan(Specifcity));
rows=unique(rows);
Specifcity(rows,:)=[];
[rows,col]=find(isnan(F1));
rows=unique(rows);
F1(rows,:)=[];

Average_Sensitivity=mean(Sensitivity)
Average_Specifcity=mean(Specifcity)
Average_F1=mean(F1)
save("metricList.mat","metricList");

