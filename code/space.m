
%%

find(strcmpi(bigram_vocab,'the_best'))

%%

A = importdata('../data/bigrams.txt','\n')

%%
train(1);

%%

t = CTimeleft(10000);
for i = 1:10000
     t.timeleft();
     a = [a train(i).bigram_idx'];
     
end
%%

a = zeros(1,size(X,2));
t = CTimeleft(numel(a));

for i = 1:numel(a)
     t.timeleft();
     a(i) = var(X(:,i));
 
     
end

%%

funlist = {@fvar3, @fvar1, @fvar}
datalist = {X(:,1:10000),X(:,10001:20000),X(:,20001:30000)}

AA = zeros(3,10000);

parfor i=1:length(funlist)
    %# call the function
    AA(i,:)=funlist{i}(datalist{i});
end
