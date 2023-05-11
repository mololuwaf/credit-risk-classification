# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

a. Explain the purpose of the analysis.
b. Explain what financial information the data was on, and what you needed to predict.
c. Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
d. Describe the stages of the machine learning process you went through as part of this analysis.
e.  Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

a. The code executes a binary classification task that aims to predict loan risk. It utilizes logistic regression, where the target variables are labeled as 0 for healthy loans and 1 for high-risk loans. Initially, the code divides the data into training and testing sets. Subsequently, it trains a logistic regression model using the original training data and assesses its performance by applying it to the testing data. To address any data imbalances, the code proceeds to resample the training data using RandomOverSampler, ensuring a balanced representation. Another logistic regression model is then trained using the resampled training data. Lastly, the code evaluates the resampled model's performance using the testing data. The evaluation metrics employed within the code encompass the balanced accuracy score, confusion matrix, and classification report.

b/c. The provided dataset contains financial details of individual borrowers who have obtained loans. It encompasses the following features:

loan_size: The amount of the loan acquired by the borrower.
interest_rate: The interest rate applied to the loan.
borrower_income: The income of the borrower.
debt_to_income: The borrower's debt-to-income ratio, calculated as the ratio of their total monthly debt payments to their monthly income.
num_of_accounts: The number of credit accounts held by the borrower.
derogatory_marks: The count of derogatory marks on the borrower's credit report.
total_debt: The overall debt amount owed by the borrower.
The objective of this dataset is to predict the loan_status of the borrower, which is a binary variable indicating whether the borrower has "Fully Paid" the loan or has "Charged Off" (defaulted on) the loan. In simpler terms, given a borrower's financial information, the task is to anticipate whether they will successfully repay their loan or default on it.

d. The machine learning process comprises various stages, such as data cleaning and preprocessing, dividing the data into features and labels, generating the training and testing datasets, selecting an appropriate model, assessing the model's performance, and comparing its performance against other models.

e. Logistics Regression and Randomoversampler

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
Based on the performance metrics, it can be observed that the second model, which is the logistic regression model fit with oversampled data, performs better than the first model. The balanced accuracy score of 0.99 is higher than the score of the first model. The confusion matrix shows only a few misclassified data points with 116 false positives and 4 false negatives. The classification report shows high precision and recall scores for both labels, indicating that the model performs well at identifying both healthy and high-risk loans. Therefore, I would recommend using the second model for predicting loan risk as it appears to be a better fit for the dataset.
