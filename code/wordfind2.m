function [idx] = wordfind2(X,Y,df_seed)


%%
%df_seed = 0.0005
idx = zeros(1,size(X,2));
thresh = int32(df_seed * size(X,1))
id = zeros(5,size(X,2));
t = CTimeleft(4);
%%
    for i=[1 2 4 5]
       %% 
        rows = find(Y==i);
        
            A = sum(X(rows,:));
            %num = (A>thresh);
            
           %%
           for j = 1:size(X,2)
            id(i,j) = A(j)>=thresh;
           end
            %%      
               
      t.timeleft();    
    end
    
    
    idx = find(sum(id)>=1);
end