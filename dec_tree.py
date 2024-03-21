# Import necessary libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import numpy as np
from C45 import C45Classifier as C45
import pandas as pd
class DecisionTreeGainRatio(DecisionTreeClassifier):
    def __init__(self, criterion="entropy", max_depth=None,
                 random_state=None, max_leaf_nodes=None):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
        )

    
    def entropy(self, y):
        """Calculate entropy."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _find_best_split(self, X, y, n_features, depth):
        """Find the best split based on gain ratio."""
        best_gain_ratio = -np.inf
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(n_features):
            values, value_counts = np.unique(X[:, feature_idx], return_counts=True)
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                mask = X[:, feature_idx] <= threshold
                y_left = y[mask]
                y_right = y[~mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calculate gain ratio for the split
                left_weight = len(y_left) / len(y)
                right_weight = len(y_right) / len(y)
                split_entropy = left_weight * self.entropy(y_left) + right_weight * self.entropy(y_right)
                info_gain = self.entropy(y) - split_entropy
                split_info = -left_weight * np.log2(left_weight) - right_weight * np.log2(right_weight)
                gain_ratio = info_gain / split_info if split_info != 0 else 0

                # Update best split if gain ratio is higher
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_gain_ratio, best_feature_idx, best_threshold
    





def generate_models(X,y):
    models=[]
    criteria_list = ['entropy', 'gini']
    for criterion in criteria_list:
        models_row=[]
        for max_depth in range(2, 11):
            clf=generate_model(criterion,X,y,max_depth)
            models_row.append(clf)
            if clf.get_depth()<max_depth:
                break
        models.append(models_row)
    clf=generate_C45_model(X,y,max_depth)
    models_row=[]
    models_row.append(clf)  
    models.append(models_row)
    return models


def generate_C45_model(X,y,max_depth=0):
    clf = C45()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X_copy = X.copy()
    for column in X_copy.columns:
        X_copy[column] = pd.cut(X_copy[column], bins=4)

    #transform interval to string
    for column in X_copy.columns:
        X_copy[column] = X[column].astype(str)
    # Iterate through each fold
    for train_index, test_index in skf.split(X_copy, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit the model
        clf.fit(X_train, y_train)
        conf_matrix, accuracy, precision, recall, f1, auc, mcc=calculate_metrics(X_test,y_test,clf)
        #print(accuracy, precision, recall, f1, auc, mcc)

    return clf     


def generate_model(criterion,X,y,max_depth):
    # Create the Decision Tree model
    if criterion=='gain ratio':
        clf = DecisionTreeGainRatio(criterion='entropy', max_depth=max_depth,random_state=42)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Iterate through each fold
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Fit the model
            clf.fit(X_train, y_train)
            conf_matrix, accuracy, precision, recall, f1, auc, mcc=calculate_metrics(X_test,y_test,clf)
            print(accuracy, precision, recall, f1, auc, mcc)

        return clf
    else:
        metrics={} 
        clf = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,random_state=42)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Iterate through each fold
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Fit the model
            clf.fit(X_train, y_train)
            conf_matrix, accuracy, precision, recall, f1, auc, mcc=calculate_metrics(X_test,y_test,clf)
            metrics
            
            print(accuracy, precision, recall, f1, auc, mcc)

        return clf 

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
