function [boost] = adaboost(X, Y, T)
% adaboost - Trains an adaboost classifier using decision stumps
%
% Usage:
%
%     [alpha h] = adaboost(X, Y, T)
%
% Returns weights and weak classifiers for every iteration. Due to decision
% stump implementation the classes need to be consecutively labeled, so
% unique(Y) = 1:n for some n. 

% Transpose for speed, as we should take columns of sparse matrices
X = X';

% For naive bayes training, Y needs to be in a different format
%Yk = bsxfun(@eq, Y, [1 2 4 5]);

% Transform Y-values to [1 2 3 4] from [1 2 4 5]
%Y(Y == 4) = 3;
%Y(Y == 5) = 4;

% Initialize Weight
n = size(Y, 1);
D = repmat(1/n, n, 1);

% Preallocate stuff
trainerrors = zeros(n,1);
h = cell(1,T);
alpha = zeros(1,T);
% Do the loop
for t = 1:T
    trainerrors = zeros(n,1);
    sampleIndices = randsample(n,n,true,D);
    %h{t} = dt_train_multi(X(:,sampleIndices)', Y(sampleIndices), 1);
    
    %%%% Naive bayes classifier as weak learner
    %h{t} = nb_train_pk(X(:,sampleIndices) > 0, Yk(sampleIndices,:));
    % Classify by argmax. Using expectation might be a good idea as
    % well.
    %[m a] = max(nb_test_pk(h{t}, X(:,sampleIndices) > 0), [], 2);
    % Trying to do expectation instead:
    %yhat = nb_test_pk(h{t}, X(:, sampleIndices) > 0);
    %yhat = round(sum(bsxfun(@times, yhat, [1 2 3 4]), 2));
    
    %%%% Liblinear 
    h{t} = liblinear_train(Y(sampleIndices), X(:,sampleIndices), '-s 7 -e 1.0', 'col');
    % use standard argmax (?) classification first
    yhat = liblinear_predict(ones(n,1), X(:,sampleIndices), h{t}, '', 'col');
    
    trainerrors = yhat ~= Y(sampleIndices);
    error = sum(D.*trainerrors);
    if error > 0.5
        T = t-1
        break;
    end
    alpha(t) = error/(1-error);
    %logicalH = bsxfun(@xor, Y, trainerrors);
    %H = ones(n,1);
    %H(logicalH == 0) = -1;
    temp = ones(n,1);
    temp(~trainerrors) = alpha(t);
    D = D.*temp;
    % Normalize
    D = D./sum(D);
    %Z = sum(D .* exp(-alpha{t}*Y_d.*H));
    %D = (D .* exp(-alpha{t}*Y_d.*H))./Z;
end

boost.h = h;
boost.alpha = alpha;