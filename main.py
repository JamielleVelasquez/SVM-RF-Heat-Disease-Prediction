# Before running this file, activate the venv using "sklearn-venv\Scripts\activate"

from concurrent.futures import process
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Using panda.io to read the dataset
unprocessed_data = pd.read_csv("heart.csv")
# print(unprocessed_data)

# Feature Selection
feature_selection = SelectKBest(chi2, k=11)
processed_data = feature_selection.fit_transform(
    unprocessed_data, unprocessed_data.loc[:, "target"])
# These features had the 3 lowest chi2 scores and will be excluded:
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results (values 0,1,2)
# number of major vessels (0-3) colored by flourosopy

#TODO: SMOTE

#TODO: RF

#TODO: SVM

#TODO: MV

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
