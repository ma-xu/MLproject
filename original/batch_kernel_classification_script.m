% Call genterate script.
clear;clc;
generate_data_script;

batch_size=150;

%% Split data
parts=ceil(size(data,1)/batch_size);
parts_end=(1:parts)'*batch_size;
parts_end(end,1)=size(data,1);

%% for plot xlim and y lim
%is 4*1 matrix. 1:x_min; 2:x_max; 3:y_min; 4:y_max;
xylim=[min(data(:,1));max(data(:,1));min(data(:,2));max(data(:,2))];

%% init predict bounder support vectors.
predict_bsv=[];
predict_abnormal=[];
Sensitivity=[];

%% For loop.
for i=1:parts
    %% Handle data
    data_batch=data(batch_size*(i-1)+1:parts_end(i,1),:);
    %data_batch=[data_batch; predict_bsv];
    
    %% Parameters for kernel SVDD
    c=0.1; % The cofficient of slack varbles.
    t=2; % kerfunction. 0linear; 1Polynomial; 2RBF; 3Sigmoid
    g=0.1;%gamma value for poly/rbf/sigmoid kernel.
    n=0.01;% paramter for oneclass svm and v-svr
    train_command=['-q -s 5',' -t ',num2str(t),' -g ',num2str(g), ' -n ',num2str(n), ' -c ',num2str(c)];
    %train_command='-s 5 -t 2 -g 0.1 -n 0.01 -c 0.1';
    
    model=libsvmtrain(data_batch(:,3),data_batch(:,1:2),train_command);
    
    %% Get bounder support vectors
    bsv_index_2=find(model.sv_coef<c);
    predict_bsv=full(model.SVs(bsv_index_2,:));
    predict_bsv=[predict_bsv ones(size(predict_bsv,1),1)];
    
    
    %% Train model and plot kernel
    pause(3);
    title_str=['batch_',num2str(i),' data'];
    [predict_label,accuracy,~]=libsvmpredict(data_batch(:,3),data_batch(:,1:2),model);
    
    % get metrics 
    metrics=myevaluate(data_batch(:,3),predict_label);
    Sensitivity=[Sensitivity;metrics.sensitivity];
    
    
    % Close previous plot windows
    close all;
    
    plot_kernel(data_batch(:,1:2),data_batch(:,end),model,xylim,title_str);
    
    
   % plot_point(data_batch(:,1:2),data_batch(:,end),xylim,title_str);
    hold on;
    
     
   
end
%mean(Sensitivity)

