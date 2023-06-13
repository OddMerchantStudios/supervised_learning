# Supervised Learning Report 

## Overview of the Analysis

* The purpose of our analysis was to predict if a loan can be classified as high-risk in an imbalanced dataset and use oversampling techniques to try and make our ML model more efficient.
* We used `lending_data.csv` which had several parameters for each loan and wheter or not it was a high-risk loan (`loan_status` column)
* Important metrics that need to be measured are `value_counts` to determine if the datasets are balanced, `accuracy_score`, `confusion matrix` and `classification report` to evaluate the different performance metrics of the models.
* The stages of the ML training process included:
  1. Reading the data into a df
  2. Separating the data into characteristics and target values (X and y)
  3. Separating that data into training and testing data
  4. Making predictions and evaluating them with accuracy scores, confusion matrix and the classification report
  5. Using oversampling to create a second set of training data
  6. Training the second model on the oversampled dataset and evaluating it
  7. Compare the two models to determine which one performed better
* For our model we used sklearn's `LogisticRegression` and from imblearn library we used `RandomOverSampler` as the oversampler of choice

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Accuracy score - 0.95
  * Precision - 100% for healthy loans and 85% for high-risk loans
  * Recall - 99% for healthy loans and 91% for high-risk loans



* Machine Learning Model 2:
  * Accuracy score - 0.99
  * Precision - 100% for healthy loans and 84% for high-risk loans
  * Recall - 99% for healthy loans and 99% for high-risk loans

## Summary
* The second ML model performed the best, as the reduction in false negatives from 56 to 4 allowed the Recall to increase from 91% to 99%, which in turn increased the accuracy score from 95% to 99%.
* In the real world, predicting high-risk loans is paramount, so having a model that can predict them as accurately as possible is definitely a priority. For that reason alone, the second option is the model of choice.

