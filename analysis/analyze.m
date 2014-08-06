function analyze( filename )


fileID = fopen(filename);
C = textscan(fileID,'%f32 %f32 %f32 %s %s %s','delimiter',' ','HeaderLines', 1);
fclose(fileID);

orig_time_rand = C{1};
time_rand = C{2};
spac_rand = C{3};
out_label = C{4};
perf_out_label = C{5};

perf_fault_num = size(find(strcmp(perf_out_label,'PFAIL')),1);
verf_fault_num = size(find(strcmp(out_label,'FAILED')),1)+size(find(strcmp(out_label,'failed')),1);
len = size(out_label,1);

time_rand1 = [];
spac_rand1 = [];
class_label = [];
time_rand2 = [];
spac_rand2 = [];
perf_class_label = [];

for i = 1 : len
    
    if ~strcmp(perf_out_label(i),'PFAIL')            
        time_rand1 = [time_rand1 time_rand(i)];
        spac_rand1 = [spac_rand1 spac_rand(i)];
        if strcmp(out_label(i),'SUCCES') || strcmp(out_label(i),'succes')
            class_label = [class_label 1];
        else
            class_label = [class_label 2];
        end
    end
    
    if ~strcmp(out_label(i),'FAILED') && ~strcmp(out_label(i),'failed')
        time_rand2 = [time_rand2 orig_time_rand(i)];
        spac_rand2 = [spac_rand2 spac_rand(i)];
        if strcmp(perf_out_label(i),'PSUCC')
            perf_class_label = [perf_class_label 1];
        else
            perf_class_label = [perf_class_label 2];
        end
    end
end

%find(perf_class_label==2);

x1 = [time_rand1' spac_rand1'];
x2 = [time_rand2' spac_rand2'];
plots(x1,class_label);
figure,hold;
plots(x2,perf_class_label);
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
