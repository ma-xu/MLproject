function [Parameters] = validate(data)
% Description: select optimal papramter for SVDD
% input: data[data label]
% output: parameters structure.
cGrid=2.^[-7:3];
tGrid=[0 1 2 3];
gGrid=2.^[-7:3];
nGrid=2.^[-7:3];
BestF1=0;

for ci=1:length(cGrid)
    for ti=1:length(tGrid)
        for gi=1:length(gGrid)
            for ni=1:length(nGrid)
                % The cofficient of slack varbles.
                c=num2str(cGrid(1,ci));
                % kerfunction. 0linear; 1Polynomial; 2RBF; 3Sigmoid
                t=num2str(tGrid(1,ti)); 
                %gamma value for poly/rbf/sigmoid kernel.
                g=num2str(gGrid(1,gi));
                % paramter for oneclass svm and v-svr
                n=num2str(nGrid(1,ni));
                command=['-q -s 5',' -t ',t,' -g ',g, ' -n ',n, ' -c ',c];
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
                    Parameters.n=nGrid(1,ni);
                    
                end 
            end
        end
    end
end

end

