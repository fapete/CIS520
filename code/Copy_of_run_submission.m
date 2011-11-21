%% Example submission: Naive Bayes

%% Load the data
clear all
load ../data/data_with_bigrams.mat;
%%
train1 = train

%% Remove Stopwords from train
% Remarks : Makes things worse. What the hell. 
words = stopwords('../data/s');
%vb = vocab;
%data = train;

train= rmstopw(train1, vocab, words);
%test = rmstopw(test,vocab,words);

%% Make the training data
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
