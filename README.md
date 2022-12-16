Improving Support Vector Machine for Heart Disease Prediction Using Synthetic Minority Oversampling Technique (SMOTE) and Confidence Voting With Random Forest Algorithm

This Python code uses the Random Forest and SVM algorithms to predict heart disease based on patient data. These algorithms were ensembled using Confidence Voting and trained on a SMOTE-treated dataset.

-------------------------------------------------------------------------------------------------------------------------------
Minimum Module Requirements: 
pandas==1.1.5
numpy==1.19.5 
matplotlib==3.2.2 
scikit_learn==0.24.2 
imbalanced_learn==0.11.0
------------------------------------------------------------------------------------------------------------------------------- 
REQUIRED DIRECTORY STRUCTURE 
… 
|__.ipynb_checkpoints 
|__ sklearn-venv
|__ jupyterTest.ipynb
|__ main.py 
|__ SMOTEData.csv
|__ heart_expanded.csv
|__ Graphs.txt
|__ heart.csv

CHANGING DATASET
In order to change the dataset used to train the models, change line 20 in main.py.

Use this line of code to use the Cleveland dataset:
unprocessed_data_X = pd.read_csv("heart.csv")

Use this line of code to use the expanded dataset:
unprocessed_data_X = pd.read_csv("heart_expanded.csv")


FEATURE SELECTION
To change feature selection parameters, change line 26 in main.py. Refer to sklearn’s documentation for possible parameter values:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

Remember to change lines 29 and 30 of main.py accordingly if you plan to change the parameter k.

SMOTE
To change SMOTE parameters, change line 41 in main.py. Refer to imblearn's documentation for possible parameter values:

https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

TRAINING TESTING SPLIT
To change the training-test split, change line 52 in main.py. The parameter "test_size" refers to the percentage of the dataset to be split for the test set. Refer to sklearn's documentation for possible parameter values:

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Random Forest
To change RF hyperparameters, change line 63 in main.py. Refer to sklearn's documentation for possible parameter values:

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

SVM
To change SVM hyperparameters, change line 93 in main.py. Refer to sklearn's documentation for possible parameter values:

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

MV (Confidence Voting)
To change MV hyperparameters, change line 119 in main.py. Refer to sklearn's documentation for possible parameter values:

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

The parameter "voting" must be kept at "hard" in order to perform majority voting (or confidence voting if there are only two models).

VISUALIZATIONS
There are multiple code snippets contained in the text file "Graphs.txt"

These code snippets can be pasted on multiple lines in main.py to visualize the data. There are comments throughout main.py that indicate the line where each code snippet is supposed to be pasted.

For example, the code snippet for the feature selection graph can be pasted into line 32.

Graphs.txt (lines 1 to 9)
# Feature Selection Graph Code

feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
plt.bar(feature_names, feature_selection.scores_)
plt.xlabel("Features")
plt.ylabel("Score")
plt.title("Feature Selection Scores")
plt.show()


main.py (line 32)
# Paste Feature Selection Graph Code Here


