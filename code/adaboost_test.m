function [Yhat] = adaboost_test(boost, Xtest)
% Generates predictions for AdaBoost on new data.
%
% Usage:
%
%   TEST_ERR MARGINS = adaboost_test(BOOST, YW_TEST, YTEST)
%
% Returns the predictions by Adaboost given a weighted combination of weak
% learners stored in the struct BOOST. YW is the predictions of the same
% pool of weak learners for the new data.

Xtest = Xtest';
n = size(Xtest,2);
Yhat = zeros(n, 1);

Yhats = zeros(n, 5);
% For every example, sum how many classifiers classify that example as 1..4
for t = 1:numel(boost.h)
    %%%% Naive Bayes
    % Classify by argmax. Might use expectation in the future.
    %[m class] = max(nb_test_pk(boost.h{t}, Xtest > 0), [], 2);
    % Trying to use expectation
    %yhat = nb_test_pk(boost.h{t}, Xtest >0);
    %class = round(sum(bsxfun(@times, yhat, [1 2 3 4]), 2));
    
    %%%% liblinear
    class = liblinear_predict(ones(n,1), Xtest, boost.h{t}, '', 'col');
    for i = 1:n
        Yhats(i, class(i)) = Yhats(i, class(i)) + log(1/boost.alpha(t));
    end
end

% Get the argmax for classification
%[m Yhat] = max(Yhats, [], 2);
% Lets try making a distribution out of every line of Yhats and then take
% the expectation of that for classification...
normalizers = sum(Yhats, 2);
Yhat = bsxfun(@rdivide, Yhats, normalizers);
Yhat = sum(bsxfun(@times, Yhat, [1 2 0 4 5]), 2);
%Yhat = round(Yhat);

% Convert results to [1 2 4 5] from [1 2 3 4]
%Yhat(Yhat == 4) = 5;
%Yhat(Yhat == 3) = 4;