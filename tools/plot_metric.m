clear;clc;
load diskmetricList.mat;

SensitivityList=[];
SpecificityList=[];
OtherSensitivity=ones(length(metricList),1)*0.921;
OtherSpecificity=ones(length(metricList),1)*0.838;
F1List=[];
for i=1:length(metricList)-1
    SensitivityList=[SensitivityList;metricList(i).sensitivity];
    SpecificityList=[SpecificityList;metricList(i).specificity];
    F1List=[F1List;metricList(i).F1_measure];
end
plot(SpecificityList,'-*r');
hold on;
plot(OtherSpecificity,'-r');
hold on;
plot(SensitivityList,'-og');
hold on;
plot(OtherSensitivity,'-g');
hold on;
plot(F1List,'-+b');
title('Metrics')
ylim([0.5,1]);
legend(["Specificity","Baseline","Sensitivity","Baseline","F1 measure"])