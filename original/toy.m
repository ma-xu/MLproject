clear
clc
rng('default');
data=rand(50,2);
y=ones(50,1);
%model=libsvmtrain(y,data,'-s 5 -t 2 -g 0.1 -n 0.001 -c 0.4');
model=libsvmtrain(y,data,'-s 5 -t 2 -g 0.1 -n 0.01 -c 0.03');
testData=rand(30,2);
plot(data(:,1),data(:,2),'.');
ylim([-1,2]);
xlim([-1,2]);
hold on;
sv=data(model.sv_indices,:);
plot(sv(:,1),sv(:,2),'ro');
hold on;
center=sum((model.sv_coef.*sv),1);
plot(center(:,1),center(:,2),'k+');
%model.sv_coef.*sv-center





% now plot support vectors
hold on;
sv = full(model.SVs);
plot(sv(:,1),sv(:,2),'ko');
% now plot decision area
[xi,yi] = meshgrid([min(data(:,1))-0.3:0.01:max(data(:,1))+0.3],[min(data(:,2))-0.3:0.01:max(data(:,2))+0.3]);
dd = [xi(:),yi(:)];
tic;
[predicted_label, accuracy, decision_values] = libsvmpredict(zeros(size(dd,1),1), dd, model);
toc
pos = find(predicted_label==1);
hold on;
redcolor = [1 0.8 0.8];
bluecolor = [1 1 1];
h1 = plot(dd(pos,1),dd(pos,2),'s','color',redcolor,'MarkerSize',10,'MarkerEdgeColor',redcolor,'MarkerFaceColor',redcolor);
pos = find(predicted_label==-1);
hold on;
h2 = plot(dd(pos,1),dd(pos,2),'s','color',bluecolor,'MarkerSize',10,'MarkerEdgeColor',bluecolor,'MarkerFaceColor',bluecolor);
uistack(h1, 'bottom');
uistack(h2, 'bottom');
