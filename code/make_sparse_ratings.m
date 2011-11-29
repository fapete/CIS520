function [X_rating] = make_sparse_ratings(Xsum, Y, rating)

%@ Param: Xsum => full(sum(X))
%@ Param: Y, an m x 1 vector of training labels
%@ Param: ratings in the set of {1, 2, 4, 5}
%->Output: 1 x n matrix, where n = number of words and
%each element is the number each word appears for a given rating,
%weighted by number of examples


Y_rating = find(Y == rating);

X_rating = zeros(size(Xsum));
for i = 1:length(Y_rating)
   X_rating(i) = Xsum(Y_rating(i));
end

weight = sum(Y == rating);

X_rating = X_rating/weight;


