Project Assessment
To assess your project work, you should be able to answer the following questions:

1. For query classification:

a. How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 100? To 1000?
Initial (no minimum threshold): 1540 unique categories
100: 879 unique categories
1000: 388 unique categories


b. What values did you achieve for P@1, R@3, and R@5? You should have tried at least a few different models, varying the minimum number of queries per
category as well as trying different fastText parameters or query normalization. Report at least 3 of your runs.
Run 1
100 minimum queries, no optinal config
N       24912
P@1     0.399
R@3     0.521
R@5     0.579


Run 2
100 minimum queries, epoch 10 learning rate 0.5 
N       24912
P@1     0.498
R@3     0.671
R@5     0.731


Run 3
1000 minimum queries, no optinal config 
N       22246
P@1     0.461
R@3     0.595
R@5     0.653


Run 4
1000 minimum queries, epoch of 15, learning rate of 0.5
N       24935
P@1     0.502
R@3     0.673
R@5     0.737


Run 5
1000 minimum queries, epoch of 25, learning rate of 0.7, wordNgrams 5
N       22246
P@1     0.536
R@3     0.703
R@5     0.768

2. For integrating query classification with search:

TODO: haven't got to this part yet!

a. Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.


b. Given 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again,include the classifier output for those queries.
