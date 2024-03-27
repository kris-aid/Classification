import sys
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.metrics import confusion_matrix,roc_curve, auc,precision_recall_curve,matthews_corrcoef

#Copiado de dec_tree.py - plots.py , no lo importe porque me daba conflicto con unas dependencias
def calculate_metrics(X,y,model):
    # Make predictions on the training set
    y_pred = model.predict(X)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    
    # Calculate true positives, false positives, false negatives
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TN = conf_matrix[0, 0]
    
    # Calculate metrics
    accuracy = (TP + TN) / np.sum(conf_matrix)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_1 = auc(fpr, tpr)
    mcc = matthews_corrcoef(y, y_pred)
    
    return conf_matrix, accuracy, precision, recall, f1, auc_1, mcc

def generate_graphs(y_test, y_pred, criterion, show = False, dataset_name = ''):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    graph_roc_curve(fpr, tpr,roc_auc,criterion= criterion, show = show  , dataset_name = dataset_name )
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    graph_precision_recall_curve(precision, recall,criterion, show = show  , dataset_name = dataset_name )

def graph_precision_recall_curve(precision, recall,criterion, show = False, dataset_name = ''):
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve', color = 'orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for criterion: '+criterion)
    plt.legend(loc="lower right")
     # Check if the directory exists, if not, create it
    directory = "neuralNetworksResults/" + dataset_name.split('.')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    plt.savefig(os.path.join(directory, f'{criterion}_PrecisionRecall.png'))
    if show:
        plt.show()
    plt.close()

def graph_roc_curve(fpr, tpr,roc_auc, criterion, show = False, dataset_name = ''):
    plt.plot(fpr, tpr, color='orange', label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for criterion: '+criterion)
    plt.legend(loc="lower right")
     # Check if the directory exists, if not, create it

    directory = "neuralNetworksResults/" + dataset_name.split('.')[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'{criterion}_ROC_curve.png'))
    
    if show:
        plt.show()
    plt.close()




#Suprimir warnings de convergencia
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # ignore convergence warnings


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


# Define stratified k-fold 
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Define parameter grid for hyperparameter tuning
param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'learning_rate_init': [0.1, 0.3, 0.5],
        'max_iter': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 200]
}

for df in dataframes:
    filename = filenames.pop(0)
    print("\n\n")
    print(filename)
    X = df.drop(columns=['Etiqueta'])  # Features (all columns except the target)
    y = df['Etiqueta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # y_mapped = y.map({0: "long events", 1: "short events"})

    # Define ANN classifier
    ann_classifier = MLPClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(ann_classifier, param_grid, cv=k_fold, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    # Print best parameters and best score
    print("Best Parameters:", best_params)

    # Crear un nuevo modelo utilizando los mejores parámetros
    best_ann_classifier = MLPClassifier(**best_params)
    # Entrenamiento
    best_ann_classifier.fit(X_train, y_train)
    y_pred = best_ann_classifier.predict(X_test)

    conf_matrix, accuracy, precision, recall, f1, auc_1, mcc = calculate_metrics(X_test, y_test, best_ann_classifier)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", auc_1)
    print("MCC:", mcc)

    # Define the directory
    directory = "neuralNetworksResults/" + filename.split('.')[0]

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save results to a text file
    txtfilename = os.path.join(directory, filename.split('.')[0] + '.txt')
    with open(txtfilename, 'w') as file:
        file.write("Best Parameters: {}\n".format(best_params))
        file.write("Confusion Matrix:\n{}\n".format(conf_matrix))
        file.write("Accuracy: {}\n".format(accuracy))
        file.write("Precision: {}\n".format(precision))
        file.write("Recall: {}\n".format(recall))
        file.write("F1 Score: {}\n".format(f1))
        file.write("AUC: {}\n".format(auc_1))
        file.write("MCC: {}\n".format(mcc))

    # Plots
    generate_graphs(y_test, y_pred, 'criterion', show = False, dataset_name = filename)
    
    