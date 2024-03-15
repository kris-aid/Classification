from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef,roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = 'subsets/subset_1_df.csv'
data_1 = pd.read_csv(dataset)
X = data_1.drop('Etiqueta', axis=1).values
x_names = data_1.drop('Etiqueta', axis=1).columns
y = data_1['Etiqueta'].values
y_names = data_1['Etiqueta'].unique()
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

from sklearn import tree

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

plt.figure(figsize=(40, 30),dpi=200)
_ = tree.plot_tree(classifiers['CART (Gini)'], 
                   feature_names=x_names,  
                   class_names=['0', '1'],
                   filled=True)
#save the tree as a png image with the name of the dataset and the classifier
plt.savefig('trees_output/tree '+ 'CART '+ 'subset 1'   +'.png')
#plt.show()
