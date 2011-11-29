%% Example submission: Naive Bayes

%% Load the data
clear all
load ../data/data_with_bigrams.mat;

%%
train1 = train
%%
train = train1
train2 = train1
%%

backt = train

%% Remove Stopwords from train
% Remarks : Makes things worse. What the hell.  
words = stopwords('../data/final.txt');

%% STRIP OFF TRAIN
t=CTimeleft(numel(train));
parfor jj=1:numel(train)
    t.timeleft();
    train(jj)=rmstopw(train(jj), vocab, words);
    train(jj)=rmstoptitle(train(jj),vocab,words);
end

%%
train = train2;

%%
train(1:2) = rmstopw(train(1:2),vocab,words);
%%
%test = rmstopw(test,vocab,words);
train = rmstoptitle(train,vocab,words);
%%
train = rmstopw(train,vocab,words)
%% Make the training data

X = make_sparse(train);
Y = double([train.rating]');
Xt = make_sparse_title(train);

%%
Xb = make_sparse_bigram(train);
%%

Xrm = make_sparse(train2);

%% Make the test data
X_test = make_sparse(test, size(X,2));
Xt_test = make_sparse_title(test, size(Xt,2));
Xb_test = make_sparse_bigram(test, size(Xb,2));

%% Find set of important unigrams and reduce the number of dimensions.
%% works good idx3 = 0.00033, idxt3 = 0.0007

%idx = wordfind1(X,Y,0.001);
%idx2 = wordfind(X,Y,0.00034);

% old version calls
%idx3 = wordfind2(X,Y,0.00002);
%idxt3 = wordfind2(Xt,Y,0.0001);

%new calls
idx3 = wordfind2(X,Y,2);
idxt3 = wordfind2(Xt,Y,3);


%%
idid = wordfind2(Xrm, Y,0.0002);
%%

Xrm1 = X(:,idid);


%%
idx = setdiff(idx3,idid);

%%
matlabpool
%%
%Remarks : good thresh for Idx3 = 0.0009, Idxt3 = 0.0006
%doesn't work. forget it.
idx3 = find(fvar(X(:,idx3))>0.001);

idxt3 = find(fvar(Xt(:,idxt3))>0.001);
idxb1 = find(fvar(Xb(:,idxb3)>2));
idxb3 = findbigrams(idx3,X,Xb,Y);

%% Determine important bigram. Need to work on it further. 0.002
%idxb3 = wordfind2(Xb,Y,0.006);
train

%%

idxb3 = wordfind2(Xb,Y,30);

%%

parfor jj = 1:size(train1,2)
train(jj) = findbigrams(train(jj),vocab,bigram_vocab,idx3);
end
%in = union(idx,idx2)
%idxbi = wordfind2(X,Y,0.005)

%%

D = [X(:,idx3) Xt(:,idxt3) Xb(:,idxb3)];


%%
D_test = [X_test(:,idx3) Xt_test(:,idxt3) Xb_test(:,idxb1)];

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

[rmse, err] = xval_error2(train, X(:,idx3), Xt(:,idxt3), Xb(:,idxb3), Y, tr_hand, te_hand);
%[rmse, err] = xval_error(train, D, Y, tr_hand, te_hand);

%% Adaboost cross validation:
% now with actually useful cross validation: Trying different values for T
% to find out which one works best. 
load ../data/data_with_bigrams.mat;

% Make the training data
X = make_sparse(train);
Y = double([train.rating]');

Xtest = make_sparse(test, size(X, 2));
%%
addpath(genpath('liblinear'));
possibleTs = 4:9;
rmse = zeros(1,numel(possibleTs));
err = zeros(1,numel(possibleTs));
i = 1;
for T = possibleTs
    tr_hand = @(X,Y) adaboost(X,Y,T);
    te_hand = @(c,x) round(adaboost_test(c,x));
    %[rmse(i), err(i)] = xval_error(train, D, Y, tr_hand, te_hand);
    [rmse(i), err(i)] = xval_error2(train, X(:,idx3), Xt(:,idxt3), Xb(:,idxb3), Y, tr_hand, te_hand);
    i = i+1;
end
%%
plot(possibleTs, rmse, possibleTs, err)
%% Adaboost xval for singular value
tr_hand = @(X,Y) adaboost_nb(X,Y,6);

te_hand = @(c,x) round(adaboost_test_nb(c,x));
%[rmse_s, err_s] = xval_error(train, D, Y, tr_hand, te_hand);
[rmse, err] = xval_error2(train, X(:,idx3), Xt(:,idxt3), Xb(:,idxb3), Y, tr_hand, te_hand);

%% Liblinear xval
tr_hand = @(X,Y) liblinear_train(Y,X, '-s 5 -e 1.0');
te_hand = @(c,x) liblinear_predict(ones(size(x,1),1), x, c);
%[rmse, err] = xval_error(train, D, Y, tr_hand, te_hand);
%[rmse, err] = xval_error2(train, X(:,idx3), Xt(:,idxt3), Xb(:,idxb3), Y, tr_hand, te_hand);
[rmse, err] = xval_error2(train, X, Xt, Xb, Y, tr_hand, te_hand);

%% k-nn xval with random projection to two dimensions
[Z, Zt] = random_projection(X,Xtest,2);
Z = full(Z);
Zt = full(Zt);
% Unfortunately my xval-function doesn't work with knn.
part = [train.category];
N = max(part);
e = zeros(1,N);
rm = zeros(1,N);

t = CTimeleft(N);
for i = 1:N
    t.timeleft();
    % Compute training set
    Di = Z(part ~= i, :);
    % Training labels
    Yi = Y(part ~= i);
    % Test fold and expected answers
    TX = Z(part == i, :);
    TY = Y(part == i);
    % Train classifier with training set
    %classifier = train_handle(Di, Yi);
    % Compute error on i'th fold
    Yhat_i = knn_test(3, Di, Yi, TX); % trying 3-nearest neighbors
    e(i) = 1/size(TX,1) * (sum(Yhat_i ~= TY));
    rm(i) = sqrt(1/size(TX,1) * sum((TY - Yhat_i).^2));
end
error = 1/double(N)*sum(e);
rmse = 1/double(N)*sum(rm);
%%



