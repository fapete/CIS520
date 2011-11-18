function [fidx fval max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

% Get entropy
H = multi_entropy(mean(Z)');

% Compute conditional entropy for each feature
ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

% Loop over each feature separately
t = CTimeleft(numel(Xrange));
fprintf('Evaluating features on %d examples: ', size(Z,1));

K = size(Z,2);

for i = 1:numel(Xrange)
    t.timeleft();

    % Check for constant values
    if numel(Xrange{i}) == 1
        ig(i) = 0; split_vals(i) = 0;
        continue;
    end
    
    % Compute all possible splits of feature
    r = linspace(Xrange{i}(1), Xrange{i}(end), min(10, numel(Xrange{i})));
    split_f = bsxfun(@le, X(:,i), r(1:end-1));
    
    % Compute conditional entropy of all possible splits
    px = mean(split_f);

    sum_x = sum(split_f);
    sum_notx =sum(~split_f);
    py_given_x = zeros(K, size(split_f,2));
    py_given_notx = zeros(K, size(split_f,2));
    for k = 1:K
        y_given_x = bsxfun(@and, Z(:,k), split_f);
        y_given_notx = bsxfun(@and, Z(:,k), ~split_f);
        
        py_given_x(k,:) = sum(y_given_x)./sum_x;
        py_given_notx(k,:) = sum(y_given_notx)./sum_notx;
    end
        
    cond_H = px.*multi_entropy(py_given_x) + ...
        (1-px).*multi_entropy(py_given_notx);
    
    % Choose split with best value
    [ig(i) best_split] = max(H-cond_H);
    split_vals(i) = r(best_split);
end

[max_ig fidx] = max(ig);
fval = split_vals(fidx);