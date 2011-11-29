

function [c] = check()
   
%%
a = []
for i = 1:size(train,2)
    
    for j = 1:size(train(i).word_idx,1)
       a = [a vocab(train(i).word_idx(j))];
       
    end
    
    if i == 1
        break;
    end
end

a

%%