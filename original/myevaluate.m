function [metrics] = myevaluate(Truelabel,Prelabel)
%METRICS Summary of this function goes here
%Author: Xu Ma. xuma@my.unt.edu
%   Truelabel: groundTruth.
%   Prelabel: predicted label
%   Truelabel and Prelabel must be -1 or 1.

% 

%   Model:
%       sensitivity:    TP/(TP+FN)  = recall\
%         
%       specificity:    TN/(TN+FP)
%       accuracy:       (TP+TN)/(TP+FP+TN+FN)
%       recall:         TP/(TP+FN)  = sensitivity
%       precision:      TP/(TP+FP)
%       F1_measure:     2PR/(P+R)=2TP/(2TP+FN+FP)
    if(length(Truelabel)~=length(Prelabel))
        error('The lengths of two labels are different.');
    end
    
    %% TP;TN;FP;FN;
    TP=sum(Prelabel(find(Truelabel==1))==1) ;
    TN=sum(Prelabel(find(Truelabel==-1))==-1); 
    FP=sum(Prelabel(find(Truelabel==1))==-1) ;
    FN=sum(Prelabel(find(Truelabel==-1))==1);
    
    %% Compute Metrics
    metrics.sensitivity=TP/(TP+FN);
    metrics.specificity=TN/(TN+FP);
    metrics.accuracy=(TP+TN)/(TP+FP+TN+FN);
    metrics.recall=TP/(TP+FN);
    metrics.precision=TP/(TP+FP);
    metrics.F1_measure=2*TP/(2*TP+FN+FP);
    
end

