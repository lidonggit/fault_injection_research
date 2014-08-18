function clustering( filename , k, fault_mark)

fileID = fopen(filename);
C = textscan(fileID,'%f32 %f32 %f32 %s %s %s','delimiter',' ','HeaderLines', 1);
fclose(fileID);

orig_time_rand = C{1};
time_rand = C{2};
spac_rand = C{3};
out_label = C{4};
perf_out_label = C{5};


fault_label = find(strcmp(out_label,fault_mark));
fault_data = [time_rand(fault_label) spac_rand(fault_label)];

idx = kmeans(fault_data(:,1),k);

plots(fault_data,idx);

xlabel('Time Randness','fontsize',18);
ylabel('Space Randness','fontsize',18);
title('Clustering','fontsize',18)
