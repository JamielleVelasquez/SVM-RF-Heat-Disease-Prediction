# Before running this file, activate the venv using "sklearn-venv\Scripts\activate"
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

import pandas as pd
import numpy as np
import sys

np.set_printoptions(suppress=True, threshold=sys.maxsize)

# Using panda.io to read the dataset
unprocessed_data_X = pd.read_csv("heart.csv")
# unprocessed_data_X = pd.read_csv("heart_expanded.csv")
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
# fasting blood sugar (fbs)

# SMOTE
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
smote = SMOTE(k_neighbors=2, sampling_strategy=0.9, random_state=1)

smote_processed_data_X, smote_processed_data_Y = smote.fit_resample(
    processed_data_X, processed_data_Y)
# Hyperparameters: sampling_strategy - 0.9 = Resample the minority to match 90% of the majority class

# 8 new entries were created by SMOTE to oversample the minority

# Paste SMOTE Scatterplot Code Here

# 70%/30% Training Test Split
processed_train_X, processed_test_X, processed_train_Y, processed_test_Y = train_test_split(
    processed_data_X, processed_data_Y, test_size=0.3, random_state=1)
smote_train_X, smote_test_X, smote_train_Y, smote_test_Y = train_test_split(
    smote_processed_data_X, smote_processed_data_Y, test_size=0.3, random_state=1)
# Hyperparameters: test_size - 30
# train_size - 70
# shuffle default = True

# RF
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

random_forest = RandomForestClassifier(bootstrap=True, max_depth=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0,
                                       n_estimators=50, criterion="log_loss", max_features="log2", min_samples_split=2, min_samples_leaf=1, random_state=12)

# RF w/ SMOTE
print('---RANDOM FOREST---')

random_forest_smote = random_forest.fit(smote_train_X, smote_train_Y)
random_forest_smote_predictions = random_forest_smote.predict(smote_test_X)
random_forest_smote_accuracy = accuracy_score(
    smote_test_Y, random_forest_smote_predictions)
print('Random Forest w/ SMOTE Accuracy: ', end="")
print(random_forest_smote_accuracy)

# RF w/o SMOTE
random_forest_processed = random_forest.fit(
    processed_train_X, processed_train_Y)
random_forest_processed_predictions = random_forest_processed.predict(
    processed_test_X)
random_forest_processed_accuracy = accuracy_score(
    processed_test_Y, random_forest_processed_predictions)
print('Random Forest w/o SMOTE Accuracy: ', end="")
print(random_forest_processed_accuracy)

# Paste RF Confusion Matrix Graph Code Here

# Paste RF Learning Curve Graph Code Here

# SVM
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

svm = SVC(kernel='rbf', C=1000, gamma=0.00001)

# SVM w/ SMOTE
svm_smote = svm.fit(smote_train_X, smote_train_Y)
svm_smote_predictions = svm_smote.predict(smote_test_X)
svm_smote_accuracy = accuracy_score(
    smote_test_Y, svm_smote_predictions)
print('---SUPPORT VECTOR MACHINE---')
print('SVM w/ SMOTE Accuracy: ', end="")
print(svm_smote_accuracy)

# SVM w/o SMOTE
svm_processed = svm.fit(processed_train_X, processed_train_Y)
svm_processed_predictions = svm_processed.predict(processed_test_X)
svm_processed_accuracy = accuracy_score(
    processed_test_Y, svm_processed_predictions)
print('SVM w/o SMOTE Accuracy: ', end="")
print(svm_processed_accuracy)

# Paste SVM Confusion Matrix Graph Code Here

# Paste SVM Learning Curve Graph Code Here

# MV
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

mv = VotingClassifier(
    estimators=[('RandomForest', random_forest), ('SVM', svm)], voting='hard')

# MV w/ SMOTE
ensemble_smote = mv.fit(smote_train_X, smote_train_Y)
ensemble_smote_predictions = ensemble_smote.predict(smote_test_X)
ensemble_smote_accuracy = accuracy_score(
    smote_test_Y, ensemble_smote_predictions)
print('---MAJORITY VOTING---')
print("MV w/ SMOTE Accuracy: ", end="")
print(ensemble_smote_accuracy)

# MV w/o SMOTE
ensemble_processed = mv.fit(processed_train_X, processed_train_Y)
ensemble_processed_predictions = ensemble_processed.predict(processed_test_X)
ensemble_processed_accuracy = accuracy_score(
    processed_test_Y, ensemble_processed_predictions)
print("MV w/o SMOTE Accuracy: ", end="")
print(ensemble_processed_accuracy)

# Paste MV Confusion Matrix Graph Code Here

# Paste MV Learning Code Graph Code Here

random_forest_smote_report = classification_report(
    smote_test_Y, random_forest_smote_predictions, output_dict=True)
random_forest_processed_report = classification_report(
    processed_test_Y, random_forest_processed_predictions, output_dict=True)

svm_smote_report = classification_report(
    smote_test_Y, svm_smote_predictions, output_dict=True)
svm_processed_report = classification_report(
    processed_test_Y, svm_processed_predictions, output_dict=True)

ensemble_smote_report = classification_report(
    smote_test_Y, ensemble_smote_predictions, output_dict=True)
ensemble_processed_report = classification_report(
    processed_test_Y, ensemble_processed_predictions, output_dict=True)

# Performance of the Models Bar Graph Code

model_labels = ['RF w/ SMOTE', 'RF w/o SMOTE', 'SVM w/ SMOTE',
                'SVM w/o SMOTE', 'MV w/ SMOTE', 'MV w/o SMOTE']

accuracy = [random_forest_smote_accuracy, random_forest_processed_accuracy, svm_smote_accuracy,
            svm_processed_accuracy, ensemble_smote_accuracy, ensemble_processed_accuracy]

precision = [random_forest_smote_report['macro avg']['precision'],
             random_forest_processed_report['macro avg']['precision'],
             svm_smote_report['macro avg']['precision'],
             svm_processed_report['macro avg']['precision'],
             ensemble_smote_report['macro avg']['precision'],
             ensemble_processed_report['macro avg']['precision']]
recall = [random_forest_smote_report['macro avg']['recall'],
          random_forest_processed_report['macro avg']['recall'],
          svm_smote_report['macro avg']['recall'],
          svm_processed_report['macro avg']['recall'],
          ensemble_smote_report['macro avg']['recall'],
          ensemble_processed_report['macro avg']['recall'],
          ]
f1_score = [random_forest_smote_report['macro avg']['f1-score'],
            random_forest_processed_report['macro avg']['f1-score'],
            svm_smote_report['macro avg']['f1-score'],
            svm_processed_report['macro avg']['f1-score'],
            ensemble_smote_report['macro avg']['f1-score'],
            ensemble_processed_report['macro avg']['f1-score'],
            ]
x = np.arange(len(model_labels))

width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(x - width - width/2, accuracy, width, label='Accuracy')
rects2 = ax.bar(x - width/2, precision, width, label='Precision')
rects3 = ax.bar(x + width/2, recall, width, label='Recall')
rects4 = ax.bar(x + width + width/2, f1_score, width, label='F1-Score')

ax.set_title('Performance of the Models')
ax.set_xticks(x, model_labels)
ax.legend()

ax.bar_label(rects1, padding=2, fmt='%.2f')
ax.bar_label(rects2, padding=2, fmt='%.2f')
ax.bar_label(rects3, padding=2, fmt='%.2f')
ax.bar_label(rects4, padding=2, fmt='%.2f')

fig.tight_layout()

plt.ylim(0.80, 0.90)
plt.show()
