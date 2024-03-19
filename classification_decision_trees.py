import pandas as pd
from dec_tree import generate_models, calculate_metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
from sklearn.model_selection import train_test_split
import os
from plots import graph_tree, generate_graphs

folder_path = 'subsets_variable_length'
dataframes=[]
filenames =[]
for filename in os.listdir(folder_path):
    filenames.append(filename)
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):  # Check if it's a file (not a directory)
        print(filename)
        df = pd.read_csv(file_path)
    dataframes.append(df)
# Create a DataFrame from the CSV file



for df in dataframes:
    filename = filenames.pop(0)
    print("\n\n")
    print(filename)
    X = df.drop(columns=['Etiqueta'])  # Features (all columns except the target)
    y = df['Etiqueta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_mapped = y.map({0: "long events", 1: "short events"})

    models=generate_models(X_train,y_train)

    criteria_list = ['entropy', 'gini','gain ratio']
    for criterion in models:
        print('///////')
        citeria = criteria_list.pop(0)
        print("  "+citeria+":")
        best_model = None
        best_auc = -1  
        best_model_conf_matrix = None
        best_model_k = None
        i = 1
        for model in criterion:
            i=i+1
            clf = model
            conf_matrix, accuracy, precision, recall, f1, auc, mcc=calculate_metrics(X_test,y_test,clf)
            print(accuracy, precision, recall, f1, auc, mcc)
            # Check if the current model has a higher AUC
            if auc > best_auc:
                best_auc = auc
                best_model = clf
                best_model_conf_matrix = conf_matrix
                best_model_k = i
        print("\nConfusion Matrix - Best Model "+str(best_model_k)+" AUC: "+str(best_auc)) 
        print(best_model_conf_matrix)
        graph_tree(best_model, X.columns, y_mapped, citeria, str(best_model_k), show=False, dataset_name=filename)

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_names= ["0", "1"]
        generate_graphs(y_test, y_pred, citeria, show=False, dataset_name=filename)
    
