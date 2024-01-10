import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import argparse

import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tabulate import tabulate 

from preprocessing import preprocess_dataset
from models import sgd, logLoss, svm, random_forest, dense_network

def parse_arguments(sys_argv): 
    print('Parsing arguments...')
    parser = argparse.ArgumentParser() 

    parser.add_argument('--model',
                        help='Select the model to use',
                        choices=['all','SGD', 'LogLoss', 'SVM', 'RF', 'DNN'],
                        default='all',
                        type=str)
    return parser.parse_args(sys_argv)

def log_results(y_val, **predictions):
    print('Logging results for the evaluation dataset:')
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
    args = parse_arguments(sys.argv[1:])

    train_df = pd.read_csv('companydata.csv')
    X_test = pd.read_csv('test_data.csv')
    y_test = pd.read_csv('test_labels.csv', header=None)
    
    X_train, y_train  = preprocess_dataset(train_df, 'train')
    X_test = preprocess_dataset(X_test, 'test')

    predictions = dict()

    match args.model:
        case 'SGD':
            predictions['SGD'] = sgd(X_train, y_train, X_test) 
        case 'LogLoss': 
            predictions['LogLoss'] = logLoss(X_train, y_train, X_test)
        case 'SVM':
            predictions['SVM'] = svm(X_train, y_train, X_test)
        case 'RF':
            predictions['Random Forest'] = random_forest(X_train, y_train, X_test)
        case 'DNN':
            predictions['Dense Network'] = dense_network(X_train, y_train, X_test)
        case 'all':        
            predictions['SGD'] = sgd(X_train, y_train, X_test) 
            predictions['Random Forest'] = random_forest(X_train, y_train, X_test)
            predictions['Dense Network'] = dense_network(X_train, y_train, X_test)
        case _:
            raise Exception(f'model:{args.model} not found')
    log_results(y_test, **predictions)
    
if __name__ == '__main__': 
    main()