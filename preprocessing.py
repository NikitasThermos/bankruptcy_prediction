from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def show_dataset_stats(df):
    print('Dataset Info:')
    print(df.info())
    print('-' * 20)
    print('Dataset Statistics:')
    print(df.describe())
    print('-' * 20)
    print('Target Column Correlations:')
    print(df.corr()['X65'].sort_values(ascending=False))

def preprocess_dataset(df, type):
    if type == 'train': 
        return df.drop('X65', axis=1), df['X65']
    else: 
        pass