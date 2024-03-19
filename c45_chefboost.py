from C45 import C45Classifier as C45
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
dataset = "subsets_variable_length/subset_1_34_df.csv"
data = pd.read_csv(dataset)
X = data.drop('Etiqueta', axis=1)
y = data['Etiqueta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clasiffier = C45()
clasiffier.fit(X_train, y_train)
y_pred = clasiffier.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
