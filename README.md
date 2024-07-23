# Credit Risk Classification

## Case Description: 
- We're working as data scientist in a risk analyst team in a finance industry and the company generates profit by giving loans to customers
- However, our company might suffer loss if the customers did not pay the loan back (we called it as default customers)
- To minimize the loss, the simple thing to do is to prevent bad applicants (who later become a default customers)
- As a data scientist, you want to create classifier model to classify good or bad applicants from the given customers data **to minimize the potential loss**

|Feature|Type|Descriptions|
|:--|:--|:--|
|`person_age`|`int`|Age|
|`person_income`|`int`|Annual Income|
|`person_home_ownership`|`str`|Home Ownership|
|`person_emp_length`|`float`|Employment length (in years)|
|`loan_intent`|`str`|Loan Intent|
|`loan_grade`|`str`|Loan Grade|
|`loan_amnt`|`int`|Loan Amount|
|`loan_int_rate`|`float`|Interest Rate|
|`loan_percent_income`|`float`|Percent Income|
|`cb_percent_default_on_file`|`str`|Historical Default|
|`cb_person_cred_hist_length`|`int`|Credit History Length|
|`loan_status`| `int` | Loan Status (0 is non default 1 is deafult) , (**our target**)|

## Workflow
1. Data Preparation
   - Load data and drop duplicates
   - Input-output Split
   - Train, test, valid Split
   - Perform EDA
   - Data Preprocessing
2. Training a Machine Learning Models
   - Define metrics
   - Define baseline model
   - Train and evaluate several model
   - Hyperparameter tuning
   - Define best model
3. Model Evaluation
    - How does best model perform>
    - Compare the financial impact

## Metrics to be used:
1. Accuracy
   -  The most basic metric, showing the percentage of the model's correct predictions.
   -  Easy to interpret, but it does not consider minority classes and does not differentiate between true positives and false positives.
2. Recall
   -  Measures the model's ability to identify borrowers who will default (true positive).
   -  Recall considers False Negatives in the evaluation process.
   -  Recall measures how well the model recognizes actual loan defaults. For lenders, missing loans that eventually default can result in significant financial losses. Therefore, achieving high Recall is crucial as it ensures most potential defaults are detected, minimizing the risk of financial setbacks.
3. Precision
   -  Measures the proportion of correct default predictions (true positives) out of all default predictions.
   - Precision considers False Positives in the evaluation process.
   -  Precision is an important metric as it evaluates the accuracy of positive predictions. Lenders need to avoid false positives (incorrectly predicting defaults) to prevent unnecessary actions. Maintaining high Precision ensures that when the model predicts a default, it is likely to be accurate. This protects lenders from unnecessary costs and maintains borrower relationships
4. F1 Score
   -  The harmonic mean of precision and recall.
   -  Useful when you want to balance both metrics.
   -  A good F1-Score indicates that our classification model has good precision and recall.
5. AUC-ROC Score
   -  Measures the model's ability to distinguish between borrowers who default and those who do not across all thresholds.
   -  Represented as a curve that shows the True Positive Rate (TPR) against the False Positive Rate (FPR) at various thresholds.

## Modelling Result
![image](https://github.com/user-attachments/assets/f4c85b14-4110-41ee-ad80-898e08f744df)

## Hyperparameter Result
![image](https://github.com/user-attachments/assets/164cc70a-13d4-4dd2-86c5-b3efc6ec77b5)

Overall, the evaluation results of the classification model show quite good performance on the test data for gradient boost (best model):
-  Accuracy: The gradient boost model has an accuracy of around 93%, indicating that it can correctly predict the majority of cases in the dataset.
-  Precision: The model has high precision for both classes, approximately 93% for the non-default class and 97% for the default class. This indicates that the model tends to make few errors when predicting a specific class.
-  Recall: The recall for the non-default class is very high, even reaching 99%, which means the model can effectively identify all non-default cases. However, the recall for the default class is slightly lower, around 74%, indicating that the model has some difficulty in identifying a portion of the default cases.
-  F1-score: A good F1-score indicates a balance between precision and recall. In this case, the F1-score for the non-default class is 96%, while for the default class, it is 84%.
-  ROC AUC Score: A high ROC AUC score of 94% indicates that the model has a strong ability to distinguish between positive and negative classes, even in cases of class imbalance.

