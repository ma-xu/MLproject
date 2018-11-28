function  plot_point(data,label,xylim,title_str)
% data must be nX2 matrix.
% label is {-1,1}.
% xylim for plot xlim and y lim
% is 4*1 matrix. 1:x_min; 2:x_max; 3:y_min; 4:y_max;

    if nargin==2
        x_max=max(data(:,1));
        x_min=min(data(:,1));
        y_max=max(data(:,2));
        y_min=min(data(:,2));
        xylim=[x_min;x_max;y_min;y_max];
        title_str='';
    elseif nargin==3
        title_str='';
    elseif size(xylim,1)~=4
        error('The length of xylim is not 4');
    end

    normal=data(find(label==1),:);
    abnormal=data(find(label~=1),:);
    plot(normal(:,1),normal(:,2),'g.');
    ylim([xylim(3,1)-1,xylim(4,1)+1]);
    xlim([xylim(1,1)-1,xylim(2,1)+1]);
    hold on;
    plot(abnormal(:,1),abnormal(:,2),'ro','MarkerSize',3);
    title(title_str);
   






end

