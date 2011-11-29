function root = build_dt(X, Y, depth_limit)
% DT_TRAIN_MULTI - Trains a multi-class decision tree classifier.
%
% Usage:
%
%    tree = dt_train(X, Y, DEPTH_LIMIT)
%
% Returns a tree of maximum depth DEPTH_LIMIT learned using the ID3
% algorithm. Assumes X is a N x D matrix of N examples with D features. Y
% must be a N x 1 {1, ..., K} vector of labels. 
%
% Returns a linked hierarchy of structs with the following fields:
%
%   node.terminal - whether or not this node is a terminal (leaf) node
%   node.fidx, node.fval - the feature based test (is X(fidx) <= fval?)
%                          associated with this node
%   node.value - 1 x K vector of P(Y==K) as predicted by this node
%   node.left - child node struct on left branch (f <= val)
%   node.right - child node struct on right branch (f > val)
%
% SEE ALSO
%    DT_CHOOSE_FEATURE_MULTI, DT_VALUE

classes = unique(Y);
K = numel(classes);
assert(isequal(unique(Y), [1:K]'), 'Y must be 1...K');

% Pre-compute the range of each feature 
for i = 1:size(X, 2)
    Xrange{i} = unique(X(:,i));
end

% Get indicator version of Y
Z = bsxfun(@eq, Y, 1:K);

root = split_node(X, Y, Z, Xrange, mean(Z), 1:size(Xrange, 2), 0, depth_limit);

function [node] = split_node(X, Y, Z, Xrange, default_value, colidx, depth, depth_limit)

% The various cases at which we will return a decision node
if depth == depth_limit || all(Y==Y(1)) || numel(Y) <= 1
    node.terminal = true;
    node.fidx = [];
    node.fval = [];
    if numel(Y) == 0
        node.value = default_value;
    else
        node.value = mean(Z,1);
    end
    node.left = []; node.right = [];

    fprintf('depth %d [%d]: Leaf node: = %s\n', depth, numel(Y), ...
        mat2str(node.value));
    return;
end

% This is not a terminal node
node.terminal = false;

% Choose a feature to split on 
[node.fidx node.fval max_ig] = dt_choose_feature_multi(X, Z, Xrange);

% Store purity of this node for posterity
node.value = mean(Z,1);

% Remove this feature from future consideration.
colidx(colidx==node.fidx) = [];

% Split the data based on this feature.
leftidx = find(X(:,node.fidx)<=node.fval);
rightidx = find(X(:,node.fidx)>node.fval);

fprintf('depth %d [%d]: Split on feature %d <= %.2f w/ IG = %.2g (L/R = %d/%d)\n', ...
    depth, numel(Y), node.fidx, node.fval, max_ig, numel(leftidx), numel(rightidx));

% Strip out this feature so we don't re-use it
node.left = split_node(X(leftidx, :), Y(leftidx), Z(leftidx,:), Xrange, node.value, colidx, depth+1, depth_limit);
node.right = split_node(X(rightidx, :), Y(rightidx), Z(rightidx, :), Xrange, node.value, colidx, depth+1, depth_limit);
