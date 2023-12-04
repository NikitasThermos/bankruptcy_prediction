import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tabulate import tabulate 

from preprocessing import show_dataset_stats, preprocess_dataset
from models import sgd, random_forest, dense_network


def load_csv():
    return pd.read_csv('companydata.csv')

def log_results(y_val, **predictions):
    print('Logging results for the evaluation dataset')
    results = []
    for model, pred in predictions.items(): 
        print(f'Confusion Matrix for {model} :')
        print(confusion_matrix(y_val, pred))
        results.append([model, recall_score(y_val, pred), 
                        precision_score(y_val, pred), f1_score(y_val, pred)])
    headers = ["Name", "Recall", "Precision", "F1"]
    table = tabulate(results, headers, tablefmt="grid")
    print(table)

def main(): 
    df = load_csv() 
    show_dataset_stats(df)
    X_train, X_val, y_train, y_val = preprocess_dataset(df)
    predictions = dict()
    predictions['SGD'] = sgd(X_train, y_train, X_val) 
    predictions['Random Forest'] = random_forest(X_train, y_train, X_val)
    predictions['Dense Network'] = dense_network(X_train, y_train, X_val, y_val)
    log_results(**predictions)
    
if __name__ == '__main__': 
    main()