%% Example submission: Naive Bayes

%% Load the data
load ../data/data_with_bigrams.mat;

% Make the training data
X = make_sparse(train);
Y = double([train.rating]');

%% Run training
Yk = bsxfun(@eq, Y, [1 2 4 5]);
nb = nb_train_pk([X]'>0, [Yk]);

%% Make the testing data and run testing
Xtest = make_sparse(test, size(X, 2));
Yhat = nb_test_pk(nb, Xtest'>0);

%% Make predictions on test set

% Convert from classes 1...4 back to the actual ratings of 1, 2, 4, 5
%[tmp, Yhat] = max(Yhat, [], 2);
ratings = [1 2 4 5];
%Yhat = ratings(Yhat)';
Yhat = sum(bsxfun(@times, Yhat, ratings), 2);
save('-ascii', 'submit.txt', 'Yhat');

%% Cross validation test/example:
ratings = [1 2 4 5];
tr_hand = @(X,Y) nb_train_pk([X]'>0, [bsxfun(@eq, Y, [1 2 4 5])]);
te_hand = @(c, x) round(sum(bsxfun(@times, nb_test_pk(c, x'>0), ratings), 2));
[rmse, err] = xval_error(train, X, Y, tr_hand, te_hand);

%% Adaboost cross validation:
% now with actually useful cross validation: Trying different values for T
% to find out which one works best. 
load ../data/data_with_bigrams.mat;

% Make the training data
X = make_sparse(train);
Y = double([train.rating]');

Xtest = make_sparse(test, size(X, 2));
addpath (genpath('liblinear'));
possibleTs = 2:7;
rmse = zeros(1,numel(possibleTs));
err = zeros(1,numel(possibleTs));
i = 1;
for T = possibleTs
    tr_hand = @(X,Y) adaboost(X,Y,T);
    te_hand = @(c,x) round(adaboost_test(c,x));
    [rmse(i), err(i)] = xval_error(train, X, Y, tr_hand, te_hand);
    i = i+1;
end
plot(possibleTs, rmse, possibleTs, err);
%% Adaboost with T = 40 xval
tr_hand = @(X,Y) adaboost(X,Y,10);
te_hand = @(c,x) round(adaboost_test(c,x));
[rmse(i), err(i)] = xval_error(train, X, Y, tr_hand, te_hand);

%% Liblinear xval
tr_hand = @(X,Y) liblinear_train(Y,X, '-s 6 -e 1.0');
te_hand = @(c,x) liblinear_predict(ones(size(x,1),1), x, c);
[rmse, err] = xval_error(train, X, Y, tr_hand, te_hand);