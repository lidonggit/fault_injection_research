fileID = fopen('CG_x.dat');
C = textscan(fileID,'%f32 %f32 %f32','delimiter',' ');
fclose(fileID);

time_rand = C{1};
spac_rand = C{2};
out_label = C{3};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = [time_rand spac_rand];
[class,type] = dbscan(x,5,[]);


%plots(x,type);
%xlabel('Time Randness','fontsize',18);
%ylabel('Space Randness','fontsize',18);
%title('Clustering')

%figure,

plots(x,out_label');
xlabel('Time Randness','fontsize',18);
ylabel('Space Randness','fontsize',18);
title('Original Label')

Num_Of_Missclassify = size(find(type~=out_label'),2);