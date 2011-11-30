%% Example submission: Naive Bayes

%% Load the data
clear all
load ../data/data_with_bigrams.mat;

%% Make the training data
X = make_sparse(train);
Y = double([train.rating]');
Xt = make_sparse_title(train, size(X,2));
Xb = make_sparse_bigram(train);
% Add features of title
XXt = X + Xt;

%% Make the test data
X_test = make_sparse(test, size(X,2));
Xt_test = make_sparse_title(test, size(Xt,2));
Xb_test = make_sparse_bigram(test, size(Xb,2));
XXt_test = X_test + Xt_test;

%% Find set of important unigrams and reduce the number of dimensions.

% Separate XXt, Xb by label and sum along m examples for all n words/bigrams
Xsum = sum(XXt);
X5 = make_sparse_ratings(Xsum, Y, 5);
X4 = make_sparse_ratings(Xsum, Y, 4);
X2 = make_sparse_ratings(Xsum, Y, 2);
X1 = make_sparse_ratings(Xsum, Y, 1);

Xsumb = sum(Xb);
X5b = make_sparse_ratings(Xsumb, Y, 5);
X4b = make_sparse_ratings(Xsumb, Y, 4);
X2b = make_sparse_ratings(Xsumb, Y, 2);
X1b = make_sparse_ratings(Xsumb, Y, 1);

% Dimension reduction:
index = non_intersect_index(X5, X4, X2, X1, 0.000003);
indexb = non_intersect_index(X5b, X4b, X2b, X1b, 0.000002);

%% Generate reduced features test and training data
D = [XXt(:,index) Xb(:,indexb)];
D_test = [XXt_test(:,index) Xb_test(:,indexb)];
%% Run training
addpath(genpath('liblinear'));
classifier = adaboost(D, Y, 14);
%% Make the testing data and run testing/make predictions
Yhat = adaboost_test(classifier, D_test);

%% Save predictions into txt-file
save('-ascii', 'submit.txt', 'Yhat');