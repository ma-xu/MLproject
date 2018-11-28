% Call genterate script.
generate_data_script;

batch_size=100;

parts=ceil(size(data,1)/batch_size);
parts_end=(1:parts)'*100;
parts_end(end,1)=size(data,1);

%for plot xlim and y lim
%is 4*1 matrix. 1:x_min; 2:x_max; 3:y_min; 4:y_max;
xylim=[min(data(:,1));max(data(:,1));min(data(:,2));max(data(:,2))];

for i=1:parts
    data_batch=data(100*(i-1)+1:parts_end(i,1),:);
    pause(3);
    title_str=['Add batch data ',num2str(i),' data'];
    %close all;
    plot_point(data_batch(:,1:2),data_batch(:,end),xylim,title_str);
     
   
end


