import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



def remove_outlier_values(dataset, threshold=2):
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(dataset)
    outliers_mask = np.abs(z_scores) > threshold
    dataset[outliers_mask] = np.nan
    return dataset

def preprocess_dataset(df, type):
    if type == 'train': 
        df = df.drop_duplicates()
        X_train, y_train = df.drop('X65', axis=1), df['X65']
        X_train = remove_outlier_values(X_train)
        return X_train, y_train
    elif type=='test': 
        X_test = remove_outlier_values(df)
        return X_test