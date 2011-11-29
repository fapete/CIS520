function [rmse, error] = xval_error2(data, X1, X2, X3, Y, train_handle, test_handle)
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

n1 = 1;
n2 = 1;%20;
n3 = 1;
nn = n1+n2+n3;
parfor i = 1:N
    %t.timeleft();
    % Compute training set
    Di1 = X1(part ~= i, :);
    Di2 = X2(part ~= i,:);
    Di3 = X3(part ~= i,:);
    
    % Training labels
    Yi = Y(part ~= i);
    % Test fold and expected answers
    TX1 = X1(part == i, :);
    TX2 = X2(part == i, :);
    TX3 = X3(part == i, :);
    TY = Y(part == i);
    %Yhat_i = zeros(size(TY,1),1);
    % Train classifier with training set
    classifier1 = train_handle(Di1, Yi);
    classifier2 = train_handle(Di2, Yi);
    class3 = train_handle(Di3,Yi);
    
    % Compute error on i'th fold
    Yhat_i1 = test_handle(classifier1, TX1);
    Yhat_i2 = test_handle(classifier2, TX2);
    Yhat_i3 = test_handle(class3, TX3);
    
    Yhat_i = (n1.*Yhat_i1 + n2.*Yhat_i2 + n3.*Yhat_i3)./nn;
    Yhat_i = round(Yhat_i);
    
    
    e(i) = 1/size(TX1,1) * (sum(Yhat_i ~= TY));
    rm(i) = sqrt(1/size(TX1,1) * sum((TY - Yhat_i).^2));
end

error = 1/double(N)*sum(e);
rmse = 1/double(N)*sum(rm);