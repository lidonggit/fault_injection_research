function [ x_min, x_max, y_min, y_max ] = bounds( x, y, label )

x_min = 1;
x_max = 0;
y_min = 1;
y_max = 0;

len = size(label,2);

for i = 1 : len
    if label(i) ~= 1
        if x(i) < x_min
            x_min = x(i);
        end
        if x(i) > x_max
            x_max = x(i);
        end
        if y(i) < y_min
            y_min = y(i);
        end
        if y(i) > y_max
            y_max = y(i);
        end
    end
end


