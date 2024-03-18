from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef,roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chefboost import Chefboost as cb

# possibles libraries for the trees implementation
# https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier
# https://github.com/serengil/chefboost
# https://github.com/tensorflow/decision-forests
# Articles to build the ID3 tree from scratch
# https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f
# https://www.geeksforgeeks.org/sklearn-iterative-dichotomiser-3-id3-algorithms/
# https://github.com/milaan9/Python_Decision_Tree_and_Random_Forest/blob/main/001_Decision_Tree_PlayGolf_ID3.ipynb


dataset = 'subsets/subset_1_df.csv'
data = pd.read_csv(dataset)
X = data.drop('Etiqueta', axis=1).values
x_names = data.drop('Etiqueta', axis=1).columns
y = data['Etiqueta'].values
y_names = data['Etiqueta'].unique()

# This part isnt working

data_modified = data.copy()
data_modified = data_modified.rename(columns={'Etiqueta': 'Decision'})
data_modified['Decision'] = data_modified['Decision'].apply(lambda x: 'Yes' if x == 1 else 'No')
#The decision colum has to be of type string, so we change the 1 and 0 to Yes and No
data_modified['Decision']=data_modified['Decision'].astype(str)

#change the etiqueta name to Decision


config_id3 = {'algorithm': 'ID3'}
config_cart = {'algorithm': 'CART'}

Tree_id3 = cb.fit(data_modified, config_id3,target_label='Decision')
# Tree_c45 = cb.fit(data, config_c45)
# Tree_cart = cb.fit(data, config_cart)


results= cb.evaluate(Tree_id3, data)
results.get('Accuracy')

# Doesnt work

# Definir los clasificadores de árbol de decisión (scikit-learn's DecisionTreeClassifier)
classifiers = {
    'CART (Gini)': DecisionTreeClassifier(criterion='gini'),
}

# Definir métricas de evaluación
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'AUC': roc_auc_score,
    'F1-Score': f1_score,
    'MCC': matthews_corrcoef
}

# Realizar Stratified 10-fold Cross-Validation y clasificación para cada árbol
results = {}

for classifier_name, classifier in classifiers.items():
    for metric_name in metrics.keys():  
        results[(classifier_name, metric_name)] = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        for metric_name, metric_func in metrics.items():
            results[(classifier_name, metric_name)].append(metric_func(y_test, y_pred))


# Calcular promedio y desviación estándar para cada métrica
averages = {}
std_devs = {}
for key, values in results.items():
    averages[key] = np.mean(values)
    std_devs[key] = np.std(values)

# Mostrar resultados de clasificación
for key in results.keys():
    print(f"Clasificador y Métrica: {key}, Promedio: {averages[key]}, Desviación Estándar: {std_devs[key]}")

# Generar matriz de confusión para cada clasificador
for classifier_name, classifier in classifiers.items():
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    print(f"Matriz de confusión para {classifier_name}:\n{cm}")



# Mostrar plot de AUC para cada clasificador
plt.figure(figsize=(10, 6))
for classifier_name in classifiers.keys():
    auc_scores = results[(classifier_name, 'AUC')]
    plt.plot(range(1, 11), auc_scores, label=classifier_name)

plt.xlabel('Fold')
plt.ylabel('AUC')
plt.title('AUC por Fold para cada Clasificador')
plt.legend()
plt.show()

# Mostrar plot de Precision vs Recall para cada clasificador
plt.figure(figsize=(10, 6))
for classifier_name in classifiers.keys():
    precision_scores = results[(classifier_name, 'Precision')]
    recall_scores = results[(classifier_name, 'Recall')]
    plt.plot(recall_scores, precision_scores, label=classifier_name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall para cada Clasificador')
plt.legend()
plt.show()
from sklearn import tree
plt.figure(figsize=(40, 30),dpi=200)
_ = tree.plot_tree(classifiers['CART (Gini)'], 
                   feature_names=x_names,  
                   class_names=['0', '1'],
                   filled=True)
#save the tree as a png image with the name of the dataset and the classifier
plt.savefig('trees_output/tree '+ 'CART '+ 'subset 1'   +'.png')
#plt.show()
