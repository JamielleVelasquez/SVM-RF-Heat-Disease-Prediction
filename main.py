# Before running this file, activate the venv using "sklearn-venv\Scripts\activate"
import scikitplot as skplt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sys

np.set_printoptions(suppress=True, threshold=sys.maxsize)

# Using panda.io to read the dataset
unprocessed_data_X = pd.read_csv("heartdisease300.csv")
unprocessed_data_y = unprocessed_data_X.loc[:, "target"]

# Feature Selection using a chi-squared scoring function
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
feature_selection = SelectKBest(chi2, k=13)
feature_selection_data = feature_selection.fit_transform(
    unprocessed_data_X, unprocessed_data_y)
processed_data_X = feature_selection_data[:, :-1]
processed_data_Y = feature_selection_data[:, 12]

# Paste Feature Selection Graph Code Here

# Hyperparameters: score_func - chi2

# This feature had the lowest chi2 scores and will be excluded:
# fasting blood sugar > 120 mg/dl

# SMOTE
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
smote_processed_data_X, smote_processed_data_Y = SMOTE(
    k_neighbors=10, random_state=2).fit_resample(processed_data_X, processed_data_Y)
# Hyperparameters: sampling_strategy - auto = resampling only the minority class
# k_neighbors = 10

# ??? new entries were created by SMOTE to oversample the minority

# 70%/30% Training Test Split
processed_train_X, processed_test_X, processed_train_Y, processed_test_Y = train_test_split(
    processed_data_X, processed_data_Y, test_size=0.3, random_state=1)
smote_train_X, smote_test_X, smote_train_Y, smote_test_Y = train_test_split(
    smote_processed_data_X, smote_processed_data_Y, test_size=0.3, random_state=1)
# Hyperparameters: test_size - 30
# train_size - 70
# shuffle default = True

# Scatterplot for unprocessed data
# print(unprocessed_data_X)
g1 = unprocessed_data_X.loc[:, "age":"target"]
plt.scatter('trestbps', 'chol', data=g1)

# Scatterplot for SMOTE treated data
# Convert SMOTE-treated data numpy array to DataFrame to use .loc
smoteDF = pd.DataFrame(smote_processed_data_X)
smoteDF.to_csv("SMOTEData.csv")

# print(smoteDF)
g2 = smoteDF.loc[1025:1051, :]  # New data from SMOTE
# print (g2)
smoteDF.plot(x=3, y=4, kind='scatter')

# RF
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

random_forest = RandomForestClassifier(bootstrap=True, max_depth=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0,
                                       n_estimators=50, criterion="log_loss", max_features="log2", min_samples_split=2, min_samples_leaf=1, random_state=50)

# RF w/ SMOTE
random_forest_smote = random_forest.fit(smote_train_X, smote_train_Y)

print('---RANDOM FOREST---')
print('Random Forest w/ SMOTE Training Set Accuracy: ', end="")
print(random_forest_smote.score(smote_train_X, smote_train_Y))
print('Random Forest w/ SMOTE Test Set Accuracy: ', end="")
print(random_forest_smote.score(smote_test_X, smote_test_Y))

# RF w/o SMOTE
random_forest_processed = random_forest.fit(
    processed_train_X, processed_train_Y)
print('Random Forest w/o SMOTE Training Accuracy: ', end="")
print(random_forest_processed.score(smote_train_X, smote_train_Y))
print('Random Forest w/o SMOTE Test Set Accuracy: ', end="")
print(random_forest_processed.score(processed_test_X, processed_test_Y))

random_forest_smote_predictions = random_forest_smote.predict(smote_test_X)
random_forest_processed_predictions = random_forest_processed.predict(
    processed_test_X)

# Paste RF Confusion Matrix Graph Code Here

# Scores for Statistics

print("\n")
print("Random Forest w/ SMOTE Classification Report")
print(classification_report(smote_test_Y, random_forest_smote_predictions))
print("Random Forest w/o SMOTE Classification Report")
print(classification_report(processed_test_Y, random_forest_processed_predictions))

# Paste RF Learning Curve Graph Code Here
