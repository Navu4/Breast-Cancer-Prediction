# Breast-Cancer-Prediction

## Objectives
This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection and hyper parameter selection. The goal is to classify whether the breast cancer is benign or malignant. To achieve this i have used machine learning classification methods to fit a function that can predict the discrete class of new input.

## Steps:
### Import Libraries
- Pandas for Data Manipulation
import pandas as pd

- Numpy for Data 
import numpy as np

- Matplotlib and Seaborn for Data Visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

- Scikit learn for different Metric, Model Selection and Classification Models

Classifiers Used for prediction Comparision:
- Logistic Regression
- Support Vector Machine (SVM)
- Navie Bayes Classifier
- Decision Tree Classifier
- K Nearest Neighbor (KNN)

### Import Dataset 
The data has been obtained from https://www.kaggle.com/. If you are interested in the social impact data can bring about, you must definitely check out their work. The Dataset is also available in this repository in the data folder. Link: https://github.com/Navu4/Breast-Cancer-Prediction/blob/master/Data/breast_cancer.csv


### Data VisualiZation using Matplotlib and Seaborn 
- Exploring the dataset
- Density and Hist plots
- Handling Missing Values


### Feature Selection 
- Correlation between the dependent and independent features using "Correlation Matrix"
- Comparing the p-values of the features and find the importance of the features 


### Splitting the dataset into Train and Test set
- Using sklearn library train_test_split

from sklearn.model_selection import train_test_split


### Create Baseline model on the Training set
- Used KFold cross validation for train the data for different train and test sets 


### Evalution of Algorithms on Standardised Data
- Created Pipeline for first Scaling the data and then training the data different classification models


### Hyperparameter Tuning - Tuning SVM
- Used GridSearchCV for hyperparameter tuning of SVM model.

from sklearn.model_selection import GridSearchCV


### Application of SVM on dataset
- Fitting the SVM model using the dataset 


### Predicting the Test set results
- Predicting the test set results using .predict and comparing the results with the actual values


### Accuracy and Evaluation of SVM Model
- Accuracy using Score
- Making the confusion matrix 
- Classification Report of ROC AUC and F1 Score value
