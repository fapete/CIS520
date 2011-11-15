function [rmse, error] = xval_error(data, X, Y, train_handle, test_handle)
% XVAL_ERROR - Computes the cross-validation error of a ML method
%
% Usage:
%
%   ERROR = xval_error(data, X, Y, train_handle, test_handle)
%
% Computes the average cross-validation error for the given classifier.
% Data is the training set (the amazon reviews), X and Y are the sparse
% matrices for input and output computed from data. train_handle is a
% function handle taking exactly two inputs: The data points and the
% corresponding outcomes. It has to return a classifier, that can then be
% used as input for test_handle. This takes two inputs: The classifier and
% the data points to be classified.

% Use categories as xval partitions
part = [data.category];
N = max(part);
e = zeros(1,N);
rm = zeros(1,N);

t = CTimeleft(N);
for i = 1:N
    t.timeleft();
    % Compute training set
    Di = X(part ~= i, :);
    % Training labels
    Yi = Y(part ~= i);
    % Test fold and expected answers
    TX = X(part == i, :);
    TY = Y(part == i);
    % Train classifier with training set
    classifier = train_handle(Di, Yi);
    % Compute error on i'th fold
    Yhat_i = test_handle(classifier, TX);
    e(i) = 1/size(TX,1) * (sum(Yhat_i ~= TY));
    rm(i) = sqrt(1/size(TX,1) * sum((TY - Yhat_i).^2));
end

error = 1/double(N)*sum(e);
rmse = 1/double(N)*sum(rm);