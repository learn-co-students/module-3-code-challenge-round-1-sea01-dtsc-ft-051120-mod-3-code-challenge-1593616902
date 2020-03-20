# Mod 3 Code Challenge: Quality Assurance

This assessment is designed to test your understanding of these areas:

1. Data Engineering
    - Importing data from a CSV
    - Handling missing values
    - Feature scaling
2. Machine Learning
    - Fitting a model on training data
    - Hyperparameter tuning
    - Model evaluation on test data

Create a new Jupyter notebook to complete the challenge and show your work. Make sure that your code is clean and readable, and that each step of your process is documented. For this challenge each step builds upon the step before it. If you are having issues finishing one of the steps completely, move on to the next step to attempt every section.  There will be occasional hints to help you move on to the next step if you get stuck, but attempt to follow the requirements whenever possible.

### Business Understanding

You have been asked by a manufacturer to build a machine learning model to help with quality assurance.  Some fraction of all parts created in their factory have manufacturing flaws, and therefore should not be shipped to customers.  The cost of processing a returned part (false negative) is relatively high, and the cost of a secondary inspection (false positive) is relatively low.  Therefore they would like a **classification** model that optimizes for **recall**.

### Data Understanding

Contained in this repo is a CSV file named `quality_assurance.csv`.  Each record represents a single part.  The columns include ten features labeled `A` through `Z`, and a target called `flawed`.  If `flawed` is equal to 1, that means that the part has a manufacturing flaw and should not be shipped to customers.

Import the quality assurance CSV data using Pandas.

### Data Preparation

#### Train-Test Split

Before performing any other preprocessing steps, split the data into training and testing sets.  Use the `train_test_split` [utility function from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) with a `random_state` of 42 for reproducibility.

#### Missing Values

At least one column in this dataset contains missing values.  Use the sklearn `SimpleImputer` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)) to fill in the missing values.  **Do not** add a "missing indicator".

If you are getting stuck at this step, you can drop the rows containing missing values, but this will mean your model performance will be worse.  To drop the rows with missing values, run:

```
df_rows_dropped = df.dropna()
```

Where `df` is the full original dataset.  Then perform your train-test split again.  (If you drop rows of X without dropping them from y, you will have a mismatch that prevents the model from fitting.)

#### Feature Scaling

Because we intend to use a model with regularization, the feature magnitudes need to be scaled in order to avoid overly penalizing the features that happen to have larger magnitudes.  Use the sklearn `StandardScaler` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)) to scale all of the features.  It is okay if you lose the feature names at this stage.

If you are getting stuck at this step, skip it.  You can increase the `max_iter` hyperparameter of the logistic regression model so it will be able to converge without scaling, although the performance will be worse.  Keep in mind whether or not you scaled the data in your final analysis.

### Modeling

#### Initial Model

Build a classification model and train it on the preprocessed training data. Use the sklearn `LogisticRegression` model ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)).

Check the performance of the logistic regression model on the training data, using the sklearn `recall_score` function ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)).  You can just compute the score, or a level-up would be to use `recall_score` within `cross_val_score` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)).  **Note:** both y-values should come from the training data; you are not using the test data yet.

If you have time, this would also be a good time to show the confusion matrix.

#### Hyperparameter Tuning

Build a second `LogisticRegression` model, this time with at least one hyperparameter changed.  (You do NOT need to use GridSearch.) Repeat the same performance checks on this model that you used for the first model.

_Hint:_ The business believes that at least one of these features is irrelevant for predicting manufacturing flaws.  Consider using hyperparameters to create a model that will perform well under these circumstances.

Write at least one sentence explaining which of the two (or more) models you are going to choose as your "best model".

### Model Evaluation

Create a new model with hyperparameters chosen based on the "best model" from the previous step.  Fit the model on the preprocessed training data.

Perform the same preprocessing steps on the test data that you performed on the training data.  Use the same `SimpleImputer` and `StandardScaler` objects fitted on the training data.

Evaluate the final model based on the test (holdout) data.  Report how it performs in terms of `recall_score`, and interpret this for a business audience.

Finally, the company suspects that at least one of the features is not actually important for predicting whether a part will have a manufacturing flaw.  Inspect the coefficients of your final model (`.coef_` attribute) and make a recommendation of whether they can save costs by no longer collecting data on one or more of the features.  (Your findings will be slightly different based on your final model choice, which is fine.)

## Summary Checklist

Before you submit your modeling analysis, make sure you have completed all of the steps:

 - [ ] Imported data, separated X and y, train-test split
 - [ ] Filled in missing values of `X_train` with a `SimpleImputer`
 - [ ] Scaled values of `X_train` with a `StandardScaler`
 - [ ] Fit a `LogisticRegression` model on `X_train` and investigated its performance in terms of `recall_score`
 - [ ] Fit a second `LogisticRegression` model on `X_train` and compared its performance to the previous model
 - [ ] Created a final `LogisticRegression` model and fit on `X_train`
 - [ ] Transformed `X_test` with the same imputer and scaler fitted on `X_train` (skip this if you did not get preprocessing finished on the training data)
 - [ ] Evaluated the final model's performance on `X_test`, and interpreted for a business audience
 - [ ] Interpreted the final model's coefficients

### Bonus: More Complex Models

NOTE: Please do not attempt this section until you have fully completed the main sections.  `git add` and `git commit` your code from the previous sections before continuing.

With any remaining time, choose a different scikit-learn classifier.  Fit it on the training data, and compare its performance on the test data to the performance of your previous "final model".

Is the `recall_score` meaningfully different?  Would your answer about dropping certain features change?
