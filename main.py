# Before running this file, activate the venv using "sklearn-venv\Scripts\activate"

from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
#import numpy as np
#np.set_printoptions(threshold=np.inf, suppress=True)

# Using panda.io to read the dataset
unprocessed_data_X = pd.read_csv("heart.csv")
unprocessed_data_y = unprocessed_data_X.loc[:, "target"]
#np.savetxt('unprocessed.csv', unprocessed_data_X, delimiter=',', fmt="%.0f")

# Feature Selection
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
feature_selection = SelectKBest(chi2, k=11)
processed_data_X = feature_selection.fit_transform(
    unprocessed_data_X, unprocessed_data_y)
#np.savetxt('no_smote.csv', processed_data_X, delimiter=',', fmt="%.0f")

# These features had the 3 lowest chi2 scores and will be excluded:
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results (values 0,1,2)
# number of major vessels (0-3) colored by flourosopy

# SMOTE
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
smote_processed_data_X, smote_processed_data_y = SMOTE(
).fit_resample(processed_data_X, processed_data_X[:, 10])
#np.savetxt('smote.csv', smote_processed_data_X, delimiter=',', fmt="%.0f")

# 27 new entries were created by SMOTE to oversample the minority

#TODO: RF
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#TODO: SVM
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#TODO: MV
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

# Pipelines

# SVM
# TODO: SVM w/ SMOTE
# TODO: SVM w/o SMOTE

# RF
# TODO: RF w/ SMOTE
# TODO: RF w/o SMOTE

# MV
# TODO: MV w/ SMOTE
# TODO: MV w/o SMOTE
