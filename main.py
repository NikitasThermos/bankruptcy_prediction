import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import pandas as pd 
import numpy as np

from preprocessing import show_dataset_stats, preprocess_dataset


def load_csv():
    return pd.read_csv('companydata.csv')


def main(): 
    df = load_csv() 
    show_dataset_stats(df)
    X_train, X_val, y_train, y_val = preprocess_dataset(df)

if __name__ == '__main__': 
    main()