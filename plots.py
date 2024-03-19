import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import auc, confusion_matrix, accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef,roc_curve
from matplotlib import pyplot as plt

def generate_graphs(y_test, y_pred, criterion):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    graph_roc_curve(fpr, tpr,roc_auc,criterion= criterion)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    graph_precision_recall_curve(precision, recall,criterion)
    

def graph_roc_curve(fpr, tpr,roc_auc, criterion):
    plt.plot(fpr, tpr, color='orange', label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for criterion: '+criterion)
    plt.legend(loc="lower right")
    plt.show()

def graph_precision_recall_curve(precision, recall,criterion):
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve', color = 'orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for criterion: '+criterion)
    plt.legend(loc="lower right")
    plt.show()
    
def graph_tree(classifier, x_names,y_names, criterion,subset_name):
    plt.figure(figsize=(40, 30),dpi=200)
    _ = plot_tree(classifier, 
                   feature_names=x_names,  
                   class_names=y_names,
                   filled=True)
    #save the tree as a png image with the name of the dataset and the classifier
    plt.savefig('trees_output/tree criterion : '+criterion+' '+ subset_name +'.png')
    plt.show()
    
    
dataset = 'subsets_variable_length/subset_1_34_df.csv'
data = pd.read_csv(dataset)
X = data.drop('Etiqueta', axis=1).values
x_names = data.drop('Etiqueta', axis=1).columns
y = data['Etiqueta'].values
y_names = data['Etiqueta'].unique()
criterion ='gini'
classifier=DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=42)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_names= ["0", "1"]
    #generate_graphs(y_test, y_pred, criterion)
    graph_tree(classifier, x_names,y_names, criterion,"subset_1_34_df")
    
#Choose the best model based on the AUC metric


# best_model = max(results, key=results.get)
# print(best_model)
# print(results[best_model])
