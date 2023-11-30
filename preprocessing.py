import pandas as pd


def show_dataset_stats(df):
    print('Dataset Info:')
    print(df.info())
    print('-' * 20)
    print('Dataset Statistics:')
    print(df.describe())
    print('-' * 20)
    print('Target Column Correlations:')
    corr_matrix = df.corr()
    print(corr_matrix['X65'].sort_values(ascending=False))