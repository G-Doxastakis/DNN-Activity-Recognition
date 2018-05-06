a = 0.8;
g = [0,0,0];
acc = table2array(WatchAccProcessed(:,1:3));
for i = 1:length(acc)
    g(1) = a*g(1) + (1-a)*acc(i,1);
    g(2) = a*g(2) + (1-a)*acc(i,2);
    g(3) = a*g(3) + (1-a)*acc(i,3);
    
    acc(i,1) = acc(i,1) - g(1);
    acc(i,2) = acc(i,2) - g(2);
    acc(i,3) = acc(i,3) - g(3);
end

WatchAccProcessed(:,1:3) = array2table(acc);
writetable(WatchAccProcessed,'WatchAccProcessed.csv')