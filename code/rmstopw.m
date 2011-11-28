function [chop] = rmstopw(data, vb, words)



%t = CTimeleft(numel(data));
for i = 1:numel(data)
    %t.timeleft();
    del =zeros(numel(data(i).word_idx),1);
    x = data(i);
    
    for j = 1:numel(data(i).word_idx)
        
        
        d = x.word_idx(j,1);
        w = vb(d);
        a = sum(strcmpi(w,words));
        if a ==0
        
            del(j) = j;
        end    
            
    end       
        del = del(del>0);
        
        data(i).word_idx(del) = [];
        data(i).word_count(del) = [];
        
end

chop = data;

end