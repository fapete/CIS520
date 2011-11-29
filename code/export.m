%%

for i = 1:size(train,2)

    if train(i).rating == 1
        fid = fopen(strcat('../data/export/1/',int2str(i),'.txt'),'w');
        for j= 1:numel(train(i).word_idx)
            ch = char(vocab(train(i).word_idx(j)));
            
            fprintf(fid,'%s ',ch);
        end
        fclose(fid);    
    end
end;

%%
fid = fopen('../data/word.txt','w');

for i = 1:size(words,2)
    ch = char(words(i));
    fprintf(fid,'%s \n',ch);
end

