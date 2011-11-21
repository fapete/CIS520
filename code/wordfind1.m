function [idx] = wordfind1(X,Y,df_seed)


%%
%df_seed = 0.001
idx = zeros(1,size(X,2));
thresh = int32(df_seed * size(X,1))
t = CTimeleft(size(X,2)/1000);
    
    for j = 1:size(X,2)
       
        f = zeros(1,4);
        for i=[1 2 4 5]
            rows = find(Y==i);
        
            A = X(rows,j);
            num = sum(A>0);
            if num(1,1)>thresh
               f(i) = 1;
            end
             
        end
        
        if sum(f)==1
                idx(j) = 1; 
        end  
    end
    idx = find(idx);
end