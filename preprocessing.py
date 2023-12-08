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

def preprocess_dataset(df):
    X_train_full, y_train_full = df.drop('X65', axis=1), df['X65']
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', MinMaxScaler())])
    X_train_full = pipeline.fit_transform(X_train_full)
    return train_test_split(X_train_full, y_train_full, test_size=0.2, 
                            stratify= y_train_full, random_state=42)