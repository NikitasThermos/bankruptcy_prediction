import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import pandas as pd 

from preprocessing import show_dataset_stats



def load_csv():
    return pd.read_csv('companydata.csv')


def main(): 
    df = load_csv() 
    show_dataset_stats(df)

if __name__ == '__main__': 
    main()