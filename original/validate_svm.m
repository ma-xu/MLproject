function [Parameters] = validate_svm(data)
% Description: select optimal papramter for SVDD
% input: data[data label]
% output: parameters structure.
cGrid=2.^[-9:2];
tGrid=[1 2];
gGrid=2.^[-9:2];

BestF1=0;
%For Ploynomal kernel
dGrid=1:5;

for ci=1:length(cGrid)
    for ti=1:length(tGrid)
        for gi=1:length(gGrid)
            % The cofficient of slack varbles.
            c=num2str(cGrid(1,ci));
            % kerfunction. 0linear; 1Polynomial; 2RBF; 3Sigmoid
            t=num2str(tGrid(1,ti)); 
            %gamma value for poly/rbf/sigmoid kernel.
            g=num2str(gGrid(1,gi));
            command=['-q',' -t ',t,' -g ',g,' -c ',c];
            model=libsvmtrain(data(:,end),data(:,1:end-1),command);
            [plabel,~,~]=libsvmpredict(data(:,end),data(:,1:end-1),model,'-q');

            % get metrics 
            metrics=myevaluate(data(:,end),plabel);
            F1=metrics.F1_measure;
            if F1>BestF1
                BestF1=F1;
                Parameters.c=cGrid(1,ci);
                Parameters.t=tGrid(1,ti);
                Parameters.g=gGrid(1,gi);
            end 
        end
    end
end

end

