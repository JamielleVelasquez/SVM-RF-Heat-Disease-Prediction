# Before running this file, activate the venv using "sklearn-venv\Scripts\activate"
import scikitplot as skplt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sys
np.set_printoptions(suppress=True, threshold=sys.maxsize)
plt.style.use('ggplot')

# Using panda.io to read the dataset
unprocessed_data_X = pd.read_csv("heart.csv")
unprocessed_data_y = unprocessed_data_X.loc[:, "target"]

# Feature Selection using a chi-squared scoring function
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

random_forest = RandomForestClassifier(n_estimators=10)

# RF w/ SMOTE
random_forest_smote = random_forest.fit(smote_train_X, smote_train_Y)
random_forest_smote_cv = cross_validate(
    random_forest_smote, smote_train_X, smote_train_Y)
print('---RANDOM FOREST---')
print('Random Forest w/ SMOTE Training Set Accuracy: ', end="")
print(np.mean(random_forest_smote_cv['test_score']))
print('Random Forest w/ SMOTE Test Set Accuracy: ', end="")
print(random_forest_smote.score(smote_test_X, smote_test_Y))

# RF w/o SMOTE
random_forest_processed = random_forest.fit(
    processed_train_X, processed_train_Y)
random_forest_processed_cv = cross_validate(
    random_forest_processed, processed_train_X, processed_train_Y)
print('Random Forest w/o SMOTE Training Accuracy: ', end="")
print(np.mean(random_forest_processed_cv['test_score']))
print('Random Forest w/o SMOTE Test Set Accuracy: ', end="")
print(random_forest_processed.score(processed_test_X, processed_test_Y))

rf_predicted_nosmote = random_forest_processed.predict(processed_test_X)
rf_predicted_smote = random_forest_smote.predict(smote_test_X)

# confusion matrix
print("confusion matrix")
print("\n")
print("without smote")
print("\n")
rf_conf_matrix_nosmote = confusion_matrix(
    processed_test_Y, rf_predicted_nosmote)
print(rf_conf_matrix_nosmote)
print("\n")
print("with smote")
print("\n")
r_conf_matrix_smote = confusion_matrix(smote_test_Y, rf_predicted_smote)
print(r_conf_matrix_smote)

# scores for stat
print("\n")
print("without smote")
print(classification_report(processed_test_Y, rf_predicted_nosmote))
print("with smote")
print(classification_report(smote_test_Y, rf_predicted_smote))

# SVM
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

svm = SVC(kernel='rbf', gamma=1,)

# SVM w/ SMOTE
svm_processed=svm.fit(smote_train_X, smote_train_Y)
svm_predicted_nosmote = svm.predict(processed_test_X)
print('---SUPPORT VECTOR MACHINE---')
print('SVM w/ SMOTE Test Set Accuracy: ', end="")
print(svm.score(smote_test_X, smote_test_Y))
print('SVM w/ SMOTE Training Set Accuracy: ', end="")
print(svm.score(smote_train_X, smote_train_Y))

# SVM w/o SMOTE
svm_smote = svm.fit(processed_data_X, processed_data_Y)
svm_predicted_smote = svm.predict(smote_test_X)

print('SVM w/o SMOTE Test Set Accuracy: ', end="")
print(svm.score(processed_test_X, processed_test_Y))
print('SVM w/o SMOTE Training Set Accuracy: ', end="")
print(svm.score(processed_train_X, processed_train_Y))
print()

#learning curve
train_sizes_abs, train_scores, test_scores = learning_curve(
    svm_smote,
    smote_train_X,
    smote_train_Y,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# plt.plot(train_sizes_abs, train_scores_mean, label="RF SMOTE-treated training set")
plt.plot(train_sizes_abs, test_scores_mean,
         label="SVM SMOTE-treated validation set")

train_sizes_abs_test, train_scores, test_scores = learning_curve(
    svm_smote,
    smote_test_X,
    smote_test_Y,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes_abs, np.mean([train_scores_mean, test_scores_mean], axis=0),
         label="SVM SMOTE-treated test set")

train_sizes_abs, train_scores, test_scores = learning_curve(
    svm_processed,
    processed_train_X,
    processed_train_Y,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
# train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# plt.plot(train_sizes_abs, train_scores_mean, label="RF Preprocessed training set")
plt.plot(train_sizes_abs, test_scores_mean,
         label="SVM Preprocessed validation set")

train_sizes_abs_test, train_scores, test_scores = learning_curve(
    svm_processed,
    processed_train_X,
    processed_train_Y,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes_abs, np.mean([train_scores_mean, test_scores_mean], axis=0),
         label="SVM Preprocessed test set")

plt.title("SVM Learning Curves")
plt.xlabel("Training Set Subset Size")
plt.ylabel("Accuracy")

plt.legend()
plt.show()

#confusion matrix
print("confusion matrix")
print("\n")
print("without smote")
print("\n")
svc_conf_matrix_nosmote = confusion_matrix(processed_test_Y,svm_predicted_nosmote)
print(svc_conf_matrix_nosmote)
print("\n")
print("with smote")
print("\n")
svc_conf_matrix_smote = confusion_matrix(smote_test_Y,svm_predicted_smote)
print(svc_conf_matrix_smote)

#scores for stat
print("\n")
print("without smote")
print(classification_report(processed_test_Y,svm_predicted_nosmote))
print("with smote")
print(classification_report(smote_test_Y,svm_predicted_smote))

# MV
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

#TODO: MV

# estimators for ensembling MV
estimators = [('RandomForest', rf), ('SVM', svm)]
ensemble_smote = VotingClassifier(estimators, voting='hard', weights=[
                                  1, 1], n_jobs=-1)  # hard voting, because we are doing MV
ensemble_smote.fit(smote_train_X, smote_train_Y)
results_smote = model_selection.cross_val_score(
    ensemble_smote, smote_train_X, smote_train_Y, scoring='accuracy')
print()
print("Validation accuracy for Ensembling w/ SMOTE: ", end="")
print(results_smote.mean())

# # TODO: Adjust MV Hyperparameters
# # Exhaustive Grid Search with Cross Validation for Optimal Hyperparameters

y_pred = ensemble_smote.predict(smote_train_X)
print(classification_report(smote_train_Y, y_pred))

skplt.metrics.plot_confusion_matrix(smote_train_Y, y_pred, figsize=(10, 8))
plt.show()

# params = {'voting':['hard'],
#           'weights':[(1,1)]}

# grid_smote = GridSearchCV(estimator=ensemble_smote, param_grid=params, cv=2)

# grid_smote.fit(smote_train_X, smote_train_Y)
# print("Best parameters for Ensembling + SMOTE: ", end="")
# print(grid_smote.best_params_)

#{'voting': 'hard', 'weights': (1, 1)}

# validation graph
param_range = np.arange(1, 10, 1, dtype=int)
train_scores, test_scores = validation_curve(
    ensemble_smote,
    smote_train_X,
    smote_train_Y,
    param_name="n_jobs",
    param_range=param_range,
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(param_range, train_scores_mean, label="train", color="blue")
plt.plot(param_range, test_scores_mean, label="test", color="red")

plt.legend()
plt.show()

ensemble_proc = VotingClassifier(estimators, voting='hard', weights=[
                                 1, 1], n_jobs=-1)  # hard voting, because we are doing MV
ensemble_proc.fit(processed_train_X, processed_train_Y)
results_proc = model_selection.cross_val_score(
    ensemble_proc, processed_train_X, processed_train_Y, scoring='accuracy')
print()
print("Validation accuracy for Ensembling w/o SMOTE: ", end="")
print(results_proc.mean())

y_pred = ensemble_proc.predict(processed_train_X)
print(classification_report(processed_train_Y, y_pred))

skplt.metrics.plot_confusion_matrix(processed_train_Y, y_pred, figsize=(10, 8))
plt.show()

# # TODO: Adjust MV Hyperparameters
# Exhaustive Grid Search with Cross Validation for Optimal Hyperparameters
# params = {'voting':['hard'],
#           'weights':[(1,1)]}

# grid_proc = GridSearchCV(estimator=ensemble_proc, param_grid=params, cv=2)

# grid_proc.fit(processed_train_X, processed_train_Y)
# print("Best parameters for Ensembling w/o SMOTE: ", end="")
# print(grid_proc.best_params_)

#{'voting': 'hard', 'weights': (1, 1)}

param_range = np.arange(1, 10, 1, dtype=int)
train_scores, test_scores = validation_curve(
    ensemble_proc,
    processed_train_X,
    processed_train_Y,
    param_name="n_jobs",
    param_range=param_range,
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(param_range, train_scores_mean, label="train", color="blue")
plt.plot(param_range, test_scores_mean, label="test", color="red")

plt.legend()
plt.show()

# # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# # Refer to MV documentation for possible parameter values
