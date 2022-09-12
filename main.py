# Before running this file, activate the venv using "sklearn-venv\Scripts\activate"

from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Using panda.io to read the dataset
unprocessed_data_X = pd.read_csv("heart.csv")
unprocessed_data_y = unprocessed_data_X.loc[:, "target"]

# Feature Selection
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
feature_selection = SelectKBest(chi2, k=11)
processed_data_X = feature_selection.fit_transform(
    unprocessed_data_X, unprocessed_data_y)
processed_data_Y = processed_data_X[:, 10]
# Hyperparameters: score_func - chi2

# These features had the 3 lowest chi2 scores and will be excluded:
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results (values 0,1,2)
# number of major vessels (0-3) colored by flourosopy

# SMOTE
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
smote_processed_data_X, smote_processed_data_Y = SMOTE(
).fit_resample(processed_data_X, processed_data_Y)
# Hyperparameters: sampling_strategy - auto = resampling only the minority class
# k_neighbors - default = 5

# 27 new entries were created by SMOTE to oversample the minority

# 70%/30% Training Test Split
processed_train_X, processed_test_X, processed_train_Y, processed_test_Y = train_test_split(
    processed_data_X, processed_data_Y, test_size=0.3)
smote_train_X, smote_test_X, smote_train_Y, smote_test_Y = train_test_split(
    smote_processed_data_X, smote_processed_data_Y, test_size=0.3)
# Hyperparameters: test_size - 30
# train_size - 70
# shuffle default = True

# RF
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
random_forest = RandomForestClassifier()
# RF w/ SMOTE
random_forest_smote = random_forest.fit(
    smote_train_X, smote_train_Y)
print('Random Forest w/ SMOTE Test Set Accuracy: ', end="")
print(random_forest_smote.score(smote_test_X, smote_test_Y))
print('Random Forest w/ SMOTE Training Set Accuracy: ', end="")
print(random_forest_smote.score(smote_train_X, smote_train_Y))
# RF w/o SMOTE
random_forest_processed = random_forest.fit(
    processed_train_X, processed_train_Y)
print('Random Forest w/o SMOTE Test Set Accuracy: ', end="")
print(random_forest_processed.score(processed_test_X, processed_test_Y))
print('Random Forest w/o SMOTE Training Set Accuracy: ', end="")
print(random_forest_processed.score(processed_train_X, processed_train_Y))
# TODO: Adjust RF Hyperparameters to avoid overfitting
# Hyperparameters: refer to documentation.

#TODO: SVM
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# SVM w/ SMOTE
# SVM w/o SMOTE
# TODO: Adjust SVM Hyperparameters
# Hyperparameters: refer to documentation.

#TODO: MV
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# MV w/ SMOTE
# MV w/o SMOTE
# TODO: Adjust MV Hyperparameters
# Hyperparameters: refer to documentation.