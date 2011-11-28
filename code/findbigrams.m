function tr = findbigrams(train,vocab,bigram_vocab,idx3)

%%
idxb = [];
A = vocab(idx3);
t = CTimeleft(size(train,2));
for i=1:size(train,2)
    t.timeleft();
    %%
     idxb=[];
      words = strtok(bigram_vocab(train(i).bigram_idx),'_')';
      
      
      
   
      for j = 1:(size(words,1)-1)
        
          if sum(strcmpi(words(j),A))>0 || sum(strcmp(words(j+1),A)) >0
              
              continue
          else
              
              idxb = [idxb; j];
          end
          
      end
      
      train(i).bigram_idx(idxb) = [];
      
      train(i).bigram_count(idxb) = [];
end



%%

tr = train;

end