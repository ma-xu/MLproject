function  plot_kernel(data,label,model,xylim,title_str)
% data must be nX2 matrix.
% label is {-1,1}.
% xylim for plot xlim and y lim
% is 4*1 matrix. 1:x_min; 2:x_max; 3:y_min; 4:y_max;

    %% Verify input parameters
    if nargin==3 
        if isempty(xylim)
            x_max=max(data(:,1));
            x_min=min(data(:,1));
            y_max=max(data(:,2));
            y_min=min(data(:,2));
        end
        title_str='';
        xylim=[x_min;x_max;y_min;y_max];
    elseif nargin==4
         title_str='';
    elseif size(xylim,1)~=4
        error('The length of xylim is not 4');
    end

    %% Plot the data.
    
    ylim([xylim(3,1)-1,xylim(4,1)+1]);
    xlim([xylim(1,1)-1,xylim(2,1)+1]);
    normal=data(find(label==1),:);
    abnormal=data(find(label~=1),:);
    plot(normal(:,1),normal(:,2),'g.','MarkerSize',10);
    
    hold on;
    plot(abnormal(:,1),abnormal(:,2),'r.','MarkerSize',10);
    title(title_str);
    
    %% Plot bounder support vectors
    bsv_index_2=find(model.sv_coef<max(model.sv_coef));
    bsv=full(model.SVs(bsv_index_2,:));
    plot(bsv(:,1),bsv(:,2),'bd','MarkerSize',10)
    
    
    
    %% plot keernel decision bounder
    % now plot support vectors
    hold on;
    sv = full(model.SVs);
    
    center=sum((model.sv_coef.*sv),1);
    plot(center(:,1),center(:,2),'k+','Linewidth',2);
    
    plot(sv(:,1),sv(:,2),'ko');
    % now plot decision area
    [xi,yi] = meshgrid([min(data(:,1))-0.3:0.01:max(data(:,1))+0.3],[min(data(:,2))-0.3:0.01:max(data(:,2))+0.3]);
    dd = [xi(:),yi(:)];
    [predicted_label, ~, ~] = libsvmpredict(zeros(size(dd,1),1), dd, model,'-q');
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
   






end

