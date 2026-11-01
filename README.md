### Why do we split a dataset into training and testing sets?
The training set teaches the model patterns in the data. A separate test set estimates how well it performs on unseen examples, helping us assess whether it generalizes rather than memorizes.

### What is the difference between classification and regression?
Classification predicts a category, such as spam or not spam. Regression predicts a continuous numeric value, such as a temperature or house price.

### Why might a Decision Tree overfit?
A tree can grow very deep and learn rules that fit individual training examples, including noise. This can lead to great training results but poor performance on new data. Limiting its depth or pruning it can help.

### What does a confusion matrix tell us that accuracy alone doesn't?
A confusion matrix shows the counts of correct and incorrect predictions for each class. It reveals which classes are confused and the types of errors, such as false positives and false negatives, that a single accuracy score can hide.

### If your model achieves 98% accuracy, does that automatically mean it is a good model?
No. If 98% of examples belong to one class, a model that always predicts that class can reach 98% accuracy while missing every example of the other class. Consider class balance, error costs, other relevant metrics, and performance on representative unseen data.

### What is the difference between a model's parameters and hyperparameters?
Parameters are learned from the training data, such as weights in a linear model. Hyperparameters are settings chosen to guide training, such as a tree's maximum depth.

### Why is reproducibility important in machine learning?
Reproducibility makes it possible to repeat experiments, verify findings, and compare approaches reliably. Tracking data and code versions, settings, and random seeds also makes results easier to debug.
