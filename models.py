import numpy as np 
from scipy.stats import randint, uniform

from sklearn.linear_model import SGDClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline

import tensorflow as tf
tf.random.set_seed(42)

def sgd(X_train, y_train, X_test):
    full_pipeline = make_pipeline(KNNImputer(weights='distance'), PolynomialFeatures(degree=2),
                                 RobustScaler(), ADASYN(random_state=42),
                                 SGDClassifier(penalty='elasticnet', random_state=42))
    
    param_distribs = {'knnimputer__n_neighbors': randint(low=10, high=500),
                  'adasyn__sampling_strategy': uniform(loc=0.1, scale=0.99),
                  'adasyn__n_neighbors': randint(low=3, high=50),
                  'sgdclassifier__class_weight': [None, 'balanced', {1: np.random.uniform(low=0.1, high=50.0)}],
                  'sgdclassifier__alpha': uniform(loc=0.0001, scale=3),
                  'sgdclassifier__loss': ['hinge', 'modified_huber', 'squared_hinge'],
                  'sgdclassifier__l1_ratio': uniform(loc=0.1, scale=0.9),
                  'sgdclassifier__learning_rate': ['optimal', 'invscaling', 'adaptive'],
                  'sgdclassifier__eta0': uniform(loc=0.0001, scale=10),

    }
    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distribs,
        n_iter=10,
        scoring='f1',
        cv=3,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f'SGD best parameters:\n {random_search.best_params_}')
    return random_search.predict(X_test)

def random_forest(X_train, y_train, X_test): 
    full_pipeline = make_pipeline(KNNImputer(), RobustScaler(), ADASYN(), PolynomialFeatures(degree=2),
                                  RandomForestClassifier(random_state=42))
    param_distibs = {
        'knnimputer__n_neighbors': randint(low=5, high=100),
        'adasyn__sampling_strategy': uniform(loc=0.1, scale=0.99),
        'adasyn__n_neighbors': randint(low=3, high=50),
        'randomforestclassifier__class_weight': [None, 'balanced'],
        'randomforestclassifier__criterion': ['gini', 'entropy'],
        'randomforestclassifier__n_estimators': randint(low=2, high=60),
    }
    
    random_search =  RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distibs,
        n_iter=5,
        scoring="f1",
        cv=5,
        verbose=2, 
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.predict(X_test)
    """
    grid_search = GridSearchCV(forest_clf, param_grid, 
                                cv=3, scoring='f1',
                                return_train_score=True, verbose=True)
    grid_search.fit(X_train, y_train)
    print('Random Forest best parameters:')
    print(grid_search.best_params_)
    return grid_search.predict(X_val)
    """

def dense_network(X_train, y_train, X_val, y_val):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_dataset = train_dataset.shuffle(buffer_size=3000).batch(32)
    val_dataset = val_dataset.batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                         monitor='val_auc',
                                                         restore_best_weights=True)
    callbacks = [early_stopping_cb]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                         beta_1=0.9, beta_2=0.999)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC(name='auc')])
    
    history = model.fit(train_dataset, epochs=30,
                        validation_data=val_dataset,
                        callbacks=callbacks)

    return [1 if p > 0.5 else 0 for p in model.predict(X_val)]