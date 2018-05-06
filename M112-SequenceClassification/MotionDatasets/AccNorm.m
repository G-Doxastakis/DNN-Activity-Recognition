g = 9.81;
acc = table2array(PhoneAccProcessed(:,1:3));
acc = acc/g;

PhoneAccProcessed(:,1:3) = array2table(acc);
writetable(PhoneAccProcessed,'PhoneAccProcessed.csv')