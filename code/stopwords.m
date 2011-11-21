function [words] = stopwords(file)
%%
%file = '../data/stopwlist.txt'

fid = fopen(file);
ch = textscan(fid,['%s \n']);

for i = 1:size(ch{1},1)
   
    words(i) = ch{1}(i);

end

end