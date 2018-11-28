%{
    Description:    
        Evaluate whole disk data without feature selection and plotting.
        Using traditional svm.
        For anomaly, we using 
    Author:         Xu Ma.
    Date:           Nov/14/2018
    Email:          xuma@my.unt.edu
%}

%% import data
data=[];
label=[];
file_dir='/home/xuma/data/Anomaly/datasets/'; % 4 types error
%file_dir='/home/xuma/data/Anomaly/disk/';
fileList={
    "20110920.mat",    
    "20110921.mat",  
    "20110922.mat",  
    "20110923.mat",  
    "20110924.mat",  
    "20110925.mat"
};

for i=1:length(fileList)
    load ([file_dir,char(fileList{1})]); % data name is Struct
    data=[data;Struct.data];
    label=[label;Struct.label];
end

%mapminmax will remap lines into a resonable scale.
data=mapminmax(data');
data=data';

clearvars -except data label;
data=[data label]; 

%% Split data
batch_size=200;
parts=ceil(size(data,1)/batch_size);
parts_end=(1:parts)'*batch_size;
parts_end(end,1)=size(data,1);

%% init predict bounder support vectors.
predict_bsv=[];
predict_abnormal=[];
Sensitivity=[];
Specifcity=[];
F1=[];
MetricList=[];

%% For loop.
for i=1:parts
    %% Handle data
    data_batch=data(batch_size*(i-1)+1:parts_end(i,1),:);
    %data_batch=[data_batch; predict_bsv];
    
   % [Parameters]=validate(data_batch)
    
    
    %% Parameters for kernel SVDD
    c=0.1; % The cofficient of slack varbles.
    t=2; % kerfunction. 0linear; 1Polynomial; 2RBF; 3Sigmoid
    g=0.1;%gamma value for poly/rbf/sigmoid kernel.
    n=0.01;% paramter for oneclass svm and v-svr
    train_command=['-q -s 5',' -t ',num2str(t),' -g ',num2str(g), ' -n ',num2str(n), ' -c ',num2str(c)];
    %train_command='-s 5 -t 2 -g 0.1 -n 0.01 -c 0.1';
    model=libsvmtrain(data_batch(:,end),data_batch(:,1:end-1),train_command);
    
    %% Get bounder support vectors
    bsv_index_2=find(model.sv_coef<c);
    predict_bsv=full(model.SVs(bsv_index_2,:));
    predict_bsv=[predict_bsv ones(size(predict_bsv,1),1)];
    
    
    %% Train model
    %pause(3);
    [predict_label,accuracy,~]=libsvmpredict(data_batch(:,end),data_batch(:,1:end-1),model,'-q');
    
    % get metrics 
    metrics=myevaluate(data_batch(:,end),predict_label);
    MetricList=[MetricList;metrics];
    Sensitivity=[Sensitivity;metrics.sensitivity];
    Specifcity=[Specifcity;metrics.specificity];
    F1=[F1;metrics.F1_measure];
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

Average_Sensitivity=mean(Sensitivity);
Average_Specifcity=mean(Specifcity);
Average_F1=mean(F1)

