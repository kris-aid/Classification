import pandas as pd
from dec_tree import generate_models, calculate_metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
from sklearn.model_selection import train_test_split
import os

folder_path = 'subsets_variable_length'
dataframes=[]
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):  # Check if it's a file (not a directory)
        print(filename)
        df = pd.read_csv(file_path)
    dataframes.append(df)
# Create a DataFrame from the CSV file

for df in dataframes:

    X = df.drop(columns=['Etiqueta'])  # Features (all columns except the target)
    y = df['Etiqueta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_mapped = y.map({0: "long events", 1: "short events"})


    models=generate_models(X_train,y_train)

    for criterion in models:
        print('///////')
        for model in criterion:
            clf = model
            conf_matrix, accuracy, precision, recall, f1, auc, mcc=calculate_metrics(X_test,y_test,clf)
            print(accuracy, precision, recall, f1, auc, mcc)

'''
# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y_mapped)
plt.show()
'''
