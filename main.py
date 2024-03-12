from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data_1 = pd.read_csv('subsets/subset_4_df.csv')
X = data_1.drop('Etiqueta', axis=1).values
y = data_1['Etiqueta'].values
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
# Métricas a evaluar
metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'mcc']
# Diccionarios para almacenar los resultados
results_knn_manhattan = {metric: [] for metric in metrics}
results_knn_euclidean = {metric: [] for metric in metrics}
results_nb = {metric: [] for metric in metrics}

nb_classifier = MultinomialNB(alpha=0.5)
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

# Puedes ajustar el parámetro k en el rango [1, 16] con pasos impares
for k in range(1, 16, 2):
    for distance in ['manhattan', 'euclidean']:
        print("\n")
        print(f"K = {k}, Distancia = {distance}")
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=distance)
        knn_classifier.fit(X_train, y_train)
        y_pred_knn = knn_classifier.predict(X_test)
        # Calcula y almacena las métricas para k-NN
        results_knn = {'accuracy': accuracy_score(y_test, y_pred_knn),
                       'precision': precision_score(y_test, y_pred_knn),
                       'recall': recall_score(y_test, y_pred_knn),
                       'roc_auc': roc_auc_score(y_test, y_pred_knn),
                       'f1': f1_score(y_test, y_pred_knn),
                       'mcc': matthews_corrcoef(y_test, y_pred_knn)}
        if distance == 'manhattan':
            for metric in metrics:
                results_knn_manhattan[metric].append(results_knn[metric])
                print(f"{metric}: {results_knn_manhattan[metric][-1]}")
            print("\n")
        elif distance == 'euclidean':
            for metric in metrics:
                results_knn_euclidean[metric].append(results_knn[metric])
                print(f"{metric}: {results_knn_euclidean[metric][-1]}")
            print("\n")
        print("\n")
        

# Print results for Manhattan distance
print("\nResultados para K-NN con distancia Manhattan:")
for metric in metrics:
    avg_metric = np.mean(results_knn_manhattan[metric])
    std_metric = np.std(results_knn_manhattan[metric])
    print(f"{metric}: Mean = {avg_metric}, Std = {std_metric}")

# Print results for Euclidean distance
print("\nResultados para K-NN con distancia Euclidiana:")
for metric in metrics:
    avg_metric = np.mean(results_knn_euclidean[metric])
    std_metric = np.std(results_knn_euclidean[metric])
    print(f"{metric}: Mean = {avg_metric}, Std = {std_metric}")

k_values = list(range(1, 16, 2))
auc_values_manhattan= results_knn_manhattan['roc_auc']
auc_values_euclidean= results_knn_euclidean['roc_auc']

def graph_k_AUC(k_values, auc_values, distance):
    plt.plot(k_values, auc_values, 'ro-')
    plt.title('k-NN: AUC vs k with '+ distance + ' distance')
    plt.xlabel('k')
    plt.xticks(k_values)
    plt.ylabel('AUC')
    plt.show()


graph_k_AUC(k_values, auc_values_manhattan, 'manhattan')
graph_k_AUC(k_values, auc_values_euclidean, 'euclidean')