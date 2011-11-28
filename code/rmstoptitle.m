function [chop] = rmstoptitle(data, vb, words)



%t = CTimeleft(numel(data));
for i = 1:numel(data)
    %t.timeleft();
    del =zeros(numel(data(i).title_idx),1);
    x = data(i);
    
    for j = 1:numel(data(i).title_idx)
        
        
        d = x.title_idx(j,1);
        w = vb(d);
        a = sum(strcmpi(w,words));
        if a ==0
        
            del(j) = j;
        end    
            
    end       
        del = del(del>0);
        
        data(i).title_idx(del) = [];
        data(i).title_count(del) = [];
        
end

chop = data;

end