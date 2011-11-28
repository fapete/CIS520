function [ss] = fvar(X)


%%
a = zeros(1,size(X,2));
t = CTimeleft(numel(a));

parfor i = 1:numel(a)
     t.timeleft();
     a(i) = var(X(:,i));
 
     
end

ss = a;
%%
end
