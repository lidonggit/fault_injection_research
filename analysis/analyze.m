function analyze( filename )


fileID = fopen(filename);
C = textscan(fileID,'%f32 %f32 %s %s','delimiter',' ');
fclose(fileID);

time_rand = C{1};
spac_rand = C{2};
out_label = C{3};

len = size(out_label,1);
class_label = ones(1,len);

for i = 1 : len
    if strcmp(out_label(i),'SUCCES') || strcmp(out_label(i),'succes')
        class_label(i) = 1;
    else
        class_label(i) = 2;
    end
end

x = [time_rand spac_rand];
plots(x,class_label);
set(gca,'XLim',[0 1]);
set(gca,'YLim',[0 1]);
xlabel('Time Randness','fontsize',18);
ylabel('Space Randness','fontsize',18);
title('Error Distribution (Black Points)','fontsize',18);







%x = [time_rand spac_rand];
%[class,type] = dbscan(x,5,[]);
%plots(x,type);
%xlabel('Time Randness','fontsize',18);
%ylabel('Space Randness','fontsize',18);
%title('Clustering')
%figure,
%Num_Of_Missclassify = size(find(type~=out_label'),2);
