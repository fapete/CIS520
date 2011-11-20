function [Z Ztest] = random_projection(X, Xtest, k)
Z = sparse(size(X,1), k);
R = normrnd(0, 1, k, size(X,2));
R = sparse(R);
Z = X*R';
Ztest = Xtest*R';
end