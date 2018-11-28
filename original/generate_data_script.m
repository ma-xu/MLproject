%generate normal data and abnormal data
rng('default');
normal_number=1000;
default_abnormal_number=100;
normal=normrnd(0,1,[normal_number,2]);
normal=[normal ones(normal_number,1)];
abnormal=normrnd(0,2,[default_abnormal_number,2]);
abnormal=[abnormal -ones(default_abnormal_number,1)];

%remove center anomaly
abnormal_dis=sum(abnormal.*abnormal,2);
[~,dis_index]=sort(abnormal_dis,'descend');

%abnormal_save_index=find(abnormal_dis>2.5);
%abnormal=abnormal(abnormal_save_index,:);

%choose the top n farest points
abnomal_number=10;
abnormal=abnormal(dis_index(1:abnomal_number),:);






data=[normal;abnormal];

% Rand index data
rand_index=randperm(size(data,1));
data=data(rand_index,:);

%Clear unnecessary variable
clearvars -except data;

%plot data
%plot_point(data(:,1:2),data(:,end));








