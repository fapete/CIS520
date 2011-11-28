function [labels] = knn(X,Y,Xt)
labels = zeros(1,size(Xt, 1));
i = 1;
for row = Xt'
    [m a] = min(sqrt(sum(bsxfun(@minus,X,row').^2,2)));
    labels(i) = Y(a);
    i = i+1;
end