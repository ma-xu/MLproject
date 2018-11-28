clear;clc;
%load /home/xuma/data/Anomaly/datasets/data.mat;
load ../data_disk.mat;
feaNum=20;
rdim=1;
c=100;
[sm sn]=size(data);
datatrain=data(1:floor(0.5*sm),:);
datatest=data(1:floor(0.5*sm),:);

[ W ] = svc(data(:,end), data(:,1:end-1),c,rdim);
[~,index]=sort(abs(W),"descend");
selected=index(1:feaNum,:)