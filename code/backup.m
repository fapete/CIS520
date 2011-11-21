for i = 1:numel(data)
    t.timeleft();
    del =zeros(numel(data(i).word_idx));
    x = data(i);
    
    for j = 1:numel(data(i).word_idx)
        
        
        d = x.word_idx(j,1);
        w = vb(d);
        a = sum(strcmp(w,words));
        if a ==1
        
            del(j) = j;
        end    
            
    end       
        del = del(del>0);
        data(i).word_idx(del) = [];
        data(i).word_count(del) = [];
       
        
        
    
    
end
