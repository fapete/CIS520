function [X_index] = non_intersect_index(X5, X4, X2, X1, margin)

X_index = [];

for i = 1: length(X5)
    if abs(X5(i) - X4(i)) > margin
        if abs(X5(i) - X2(i)) > margin
            if abs(X5(i) - X1(i)) > margin
                if abs(X4(i) - X2(i)) > margin
                    if abs(X4(i) - X1(i)) > margin
                        if abs(X2(i) - X1(i)) > margin
                            X_index = horzcat(X_index, i);
                        end
                    end
                end
            end
        end
    end
end

                
        