from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef,auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Métricas a evaluar
metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'mcc']
k_values = list(range(1, 16, 2))

def graph_k_AUC(k_values, auc_values, distance):
    plt.plot(k_values, auc_values, 'ro-')
    plt.title('k-NN: AUC vs k with '+ distance + ' distance')
    plt.xlabel('k')
    plt.xticks(k_values)
    plt.ylabel('AUC')
    plt.show()

def classify(dataset,verbose=False): 
    data_1 = pd.read_csv(dataset)
    X = data_1.drop('Etiqueta', axis=1).values
    y = data_1['Etiqueta'].values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
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

    if verbose:
        print("\nResultados para Naive Bayes:")
        for metric in metrics:
            print(f"{metric}: Mean = {avg_results_nb[metric][0]}, Std = {avg_results_nb[metric][1]}")

    # from sklearn.neighbors import KNeighborsClassifier

    best_k = []

    # Puedes ajustar el parámetro k en el rango [1, 16] con pasos impares
    for k in k_values:
        for distance in ['manhattan', 'euclidean']:
            if verbose:
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
                    if verbose:
                        print(f"{metric}: {results_knn_manhattan[metric][-1]}")
                if verbose:
                    print("\n")
            elif distance == 'euclidean':
                for metric in metrics:
                    results_knn_euclidean[metric].append(results_knn[metric])
                    if verbose:
                        print(f"{metric}: {results_knn_euclidean[metric][-1]}")
                if verbose:
                    print("\n")
                    print("\n")

        #Comparar AUC de manhattan y euclidean y obtener el mejor k
        knn_manhattan_auc = results_knn_manhattan['roc_auc']
        knn_euclidean_auc = results_knn_manhattan['roc_auc']
        max_value_manhattan = max(knn_manhattan_auc)
        max_index_manhattan = knn_manhattan_auc.index(max_value_manhattan)
        max_index_manhattan = max_index_manhattan * 2 + 1

        max_value_euclidean = max(knn_euclidean_auc)
        max_index_euclidean = knn_euclidean_auc.index(max_value_euclidean)
        max_index_euclidean = max_index_euclidean * 2 + 1
        
        if max_value_manhattan > max_value_euclidean:
            best_k= [max_value_manhattan, max_index_manhattan, 'manhattan']
        else:
            best_k= [max_value_euclidean, max_index_euclidean, 'euclidean']
        
        
    if verbose:
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
        
    return results_nb, results_knn_manhattan, results_knn_euclidean, best_k

if __name__ == "__main__":
    
    subsets = ['subsets/subset_1_df.csv', 'subsets/subset_2_df.csv', 'subsets/subset_3_df.csv', 'subsets/subset_4_df.csv', 'subsets/subset_5_df.csv']
    total_results_nb = {metric: [] for metric in metrics}
    total_results_knn_manhattan = {metric: [] for metric in metrics}
    total_results_knn_euclidean = {metric: [] for metric in metrics}
    all_best_k = []
    mean_auc_values_manhattan = []
    mean_auc_values_euclidean = []

    counter = 0
    for subset in subsets:
        results_nb, results_knn_manhattan, results_knn_euclidean, results_best_k = classify(subset, False)
        all_best_k.append(results_best_k)
        for metric in metrics:
            total_results_nb[metric].extend(results_nb[metric])
            total_results_knn_manhattan[metric].extend(results_knn_manhattan[metric])
            total_results_knn_euclidean[metric].extend(results_knn_euclidean[metric])

        if counter == 0:
            mean_auc_values_manhattan = results_knn_manhattan['roc_auc']
            mean_auc_values_euclidean = results_knn_euclidean['roc_auc']
        else:
            mean_auc_values_manhattan = [((x + y)/2) for x, y in zip(mean_auc_values_manhattan, results_knn_manhattan['roc_auc'])]
            mean_auc_values_euclidean = [((x + y)/2) for x, y in zip(mean_auc_values_euclidean, results_knn_euclidean['roc_auc'])]
        counter += 1

    mean_results_nb = {metric: np.mean(total_results_nb[metric]) for metric in metrics}
    mean_results_knn_manhattan = {metric: np.mean(total_results_knn_manhattan[metric]) for metric in metrics}
    mean_results_knn_euclidean = {metric: np.mean(total_results_knn_euclidean[metric]) for metric in metrics}
    
    print("\nMean Results for Naive Bayes:")
    for metric in metrics:
        print(f"{metric}: Mean = {mean_results_nb[metric]}")

    print("\nMean Results for K-NN with Manhattan distance:")
    for metric in metrics:
        print(f"{metric}: Mean = {mean_results_knn_manhattan[metric]}")

    print("\nMean Results for K-NN with Euclidean distance:")
    for metric in metrics:
        print(f"{metric}: Mean = {mean_results_knn_euclidean[metric]}")

    print("\nBest k for each subset:")
    for i in range(len(all_best_k)):
        print(f"Subset {i+1}: {all_best_k[i]}")

    graph_k_AUC(k_values, mean_auc_values_manhattan, 'manhattan')
    graph_k_AUC(k_values, mean_auc_values_euclidean, 'euclidean')