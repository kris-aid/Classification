from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB


data_1 = pd.read_csv('subsets/subset_1_df.csv')
X = data_1.drop('Etiqueta', axis=1).values
y = data_1['Etiqueta'].values
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
# Métricas a evaluar
metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'mcc']
# Diccionarios para almacenar los resultados
results_knn = {metric: [] for metric in metrics}
results_nb = {metric: [] for metric in metrics}

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Calcula y almacena las métricas para Naive Bayes
results_nb['accuracy'].append(accuracy_score(y_test, y_pred_nb))
results_nb['precision'].append(precision_score(y_test, y_pred_nb))
results_nb['recall'].append(recall_score(y_test, y_pred_nb))
results_nb['roc_auc'].append(roc_auc_score(y_test, y_pred_nb))
results_nb['f1'].append(f1_score(y_test, y_pred_nb))
results_nb['mcc'].append(matthews_corrcoef(y_test, y_pred_nb))

avg_results_nb = {metric: (np.mean(results_nb[metric]), np.std(results_nb[metric])) for metric in metrics}

print("\nResultados para Naive Bayes:")
for metric in metrics:
    print(f"{metric}: Mean = {avg_results_nb[metric][0]}, Std = {avg_results_nb[metric][1]}")

# from sklearn.neighbors import KNeighborsClassifier

# # Puedes ajustar el parámetro k en el rango [1, 16] con pasos impares
# for k in range(1, 16, 2):
#     knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # o 'manhattan'
#     knn_classifier.fit(X_train, y_train)
#     accuracy_knn = knn_classifier.score(X_test, y_test)
#     # Realiza la evaluación y optimización según sea necesario
