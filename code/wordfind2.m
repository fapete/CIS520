function [idx] = wordfind2(X,Y,df_seed)


%%
%df_seed = 0.0005
idx = zeros(1,size(X,2));

%thresh = df_seed * size(X,1);
thresh = df_seed
id = zeros(5,size(X,2));
t = CTimeleft(4);
%%
    for i=[1 2 4 5]
       %%
       t.timeleft();    
        %rows = find(Y==i);
        
            A = sum(X(Y==i,:));
            %num = (A>thresh);
            
           %%
           parfor j = 1:size(X,2)
            id(i,j) = A(j)>=thresh;
            id1(i,j) = A(j);
           end
            %%      
               
    end
    
    %%
    idx = find(sum(id)>=1);
end