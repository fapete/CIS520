%% Example submission: Naive Bayes

%% Load the data
clear all
load ../data/data_with_bigrams.mat;

%% Make the training data
X = make_sparse(train);
Y = double([train.rating]');
Xt = make_sparse_title(train);
Xb = make_sparse_bigram(train);

%% Make the test data
X_test = make_sparse(test, size(X,2));
Xt_test = make_sparse_title(test, size(Xt,2));
Xb_test = make_sparse_bigram(test, size(Xb,2));

%% Find set of important unigrams and reduce the number of dimensions.

idx3 = wordfind2(X,Y,0.00033);
idxt3 = wordfind2(Xt,Y,0.0006);
%%

idxb3 = findbigrams(idx3,X,Xb,Y);


%% Determine important bigrams. Need to work on it further.
idxb3 = wordfind2(Xb,Y,0.0005);

%% Generate reduced features test and training data
D = [X(:,idx3) Xt(:,idxt3) Xb(:,idxb3)];
D_test = [X_test(:,idx3) Xt_test(:,idxt3) Xb_test(:,idxb3)];
%% Run training
addpath(genpath('liblinear'));
classifier = adaboost(D, Y, 5);
%% Make the testing data and run testing/make predictions
Yhat = adaboost_test(classifier, D_test);

%% Save predictions into txt-file
save('-ascii', 'submit.txt', 'Yhat');
