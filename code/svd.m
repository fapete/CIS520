clear ('-regexp', '^X');

%% Generate relevant data matrices
X = make_sparse(train);
Xt = make_sparse_title(train, size(X,2));
Xb = make_sparse_bigram(train);

%Xtest = make_sparse(test);
%XtestT = make_sparse_title(test, size(Xtest,2));

%Xtestb = make_sparse_bigram(test);

Y = double([train.rating]');

%% Add features of title to Examples

Xxt = X + Xt;
%XtestXT = Xtest + XtestT;
%% Separate Xxt, Xb by label and sum along m examples for all n words/bigrams
Xsum = (sum(Xxt));
X5 = make_sparse_ratings(Xsum, Y, 5);
X4 = make_sparse_ratings(Xsum, Y, 4);
X2 = make_sparse_ratings(Xsum, Y, 2);
X1 = make_sparse_ratings(Xsum, Y, 1);

Xsumb = (sum(Xb));
X5b = make_sparse_ratings(Xsumb, Y, 5);
X4b = make_sparse_ratings(Xsumb, Y, 4);
X2b = make_sparse_ratings(Xsumb, Y, 2);
X1b = make_sparse_ratings(Xsumb, Y, 1);

%% Dimension reduction: find index of words that appear beyond a margin of 0 in all Xi's.
index = non_intersect_index(X5, X4, X2, X1, 0.000001);
indexb = non_intersect_index(X5b, X4b, X2b, X1b, 0.000007);
%% Either...Form reduced training and testing matrix (best results so far w/o adaboost)
smallXxt = Xxt(:,index);
smallXb =(Xb(:,indexb));

%% SVD
[U S V] = pca(smallXxt, 200);
[Ub Sb Vb] = pca(smallXb, 200);

sXR = smallXxt * V;
sXbR = smallXb *Vb;

smallX = [sXR, sXbR];
smallX = sparse(smallX);

%Xtestb = Xtestb(:, indexb);

%smallX = [Xxt(:, index), Xb];
%smallXtest = [XtestXT(:, index), Xtestb];

%% Or, Form reduced training and testing matrix
%smallX = Xxt(:, index);
%smallXtest = XtestXT(:, index);

%% Or, Form reduced training and testing matrix, applying tfidf alogrithm
%smallX = tfidf(Xxt(:, index));
%smallXtest = tfidf(XtestXT(:, index));
%% Clear some stuff to free up workspace 
%clear ('-regexp', '^X5|^X4|^X2|^X1');
clear ('-regexp', '^X');
clear ('-regexp', '^index|^diff|^zero');

%% Either....Run liblinear on reduced matrices 
tr_hand = @(smallX,Y) liblinear_train(Y,smallX, '-s 0 -e 0.1');
te_hand = @(c,smallX) liblinear_predict(ones(size(smallX,1),1), smallX, c);
[rmse, err] = xval_error_fabian(train, smallX, Y, tr_hand, te_hand);

%te_hand = @(c,smallXtest) liblinear_predict(ones(size(smallXtest,1),1), smallXtest, c);

%when smallX is Xxt@Xb (rmse: 1.0752 , err: 0.3083); s6, T = 5,
%marginX = 0.000003; marginB = 0; using smallX in te_hand



%% Or... Run adaboost on reduced matrices. {weak learner: liblinear; T =5}
%tr_hand = @(smallX,Y) adaboost_fabian(smallX,Y, 5);
%te_hand = @(boost,x) round(adaboost_test_fabian(boost,x));
%[rmse, err] = xval_error_fabian(train, smallX, Y, tr_hand, te_hand);
%when smallX is Xxt @ (Xb) (rmse: 0.9856 , err: 0.3620); margins = 0; T =5;
%s7 <--- best so far

%when smallX is Xxt @ (Xb) (rmse: 0.9968 , err: 0.3591); marginX =
%0.000003; marginB = 0.000003; S7;

%when smallX is Xxt @ (Xb) (rmse: 1.0028 , err: 0.3581); marginX =
%0.000003; marginB = 0; S7;

%when smallX is Xxt@Xb (rmse: 1.0352 , err: 0.3851); s6, T = 5,
%margin=0 

%when smallX is Xxt@Xb (rmse: 1.0346 , err: 0.3902); s6, T = 5,
%marginX = 0.000003; marginB = 0; <---best w/ s6 so far

