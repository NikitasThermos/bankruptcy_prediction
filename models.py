
import joblib

import numpy as np 
from scipy.stats import randint, uniform

from sklearn.linear_model import SGDClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline

import tensorflow as tf
tf.random.set_seed(42)


def logLoss(X_train, y_train, X_test, args):
    if args.best_parameters: 
        model = joblib.load('parameters/logloss.pki')
        return model.predict(X_test)
     
    full_pipeline = make_pipeline(KNNImputer(weights='distance'), PolynomialFeatures(degree=2, include_bias=False),
                                RobustScaler(), ADASYN(sampling_strategy='minority', random_state=42),
                                SGDClassifier(loss='log_loss', learning_rate='adaptive', 
                                                penalty='elasticnet',  class_weight=None, random_state=42))

    param_distribs = {'knnimputer__n_neighbors': randint(low=10, high=500),
                    'adasyn__n_neighbors': randint(low=3, high=100),
                    'sgdclassifier__alpha': uniform(loc=0.0001, scale=3),
                    'sgdclassifier__l1_ratio': uniform(loc=0.1, scale=0.9),
                    'sgdclassifier__eta0': uniform(loc=0.0001, scale=10),
    }

    random_search = RandomizedSearchCV(
            full_pipeline,
            param_distributions=param_distribs,
            n_iter=10,
            scoring='f1',
            cv=3,
            verbose=4,
            random_state=42,
    )

    random_search.fit(X_train, y_train)
    if args.save_model: 
        joblib.dump(random_search, 'parameters/logloss.pki')
    return random_search.predict(X_test)

def svm(X_train, y_train, X_test, args):
    if args.best_parameters:
        model = joblib.load('parameters/svm_poly.pki')
        return model.predict(X_test)
    
    full_pipeline = make_pipeline(KNNImputer(weights='distance'), ADASYN(sampling_strategy='minority'),  
                                  RobustScaler(),  SVC(kernel='poly', degree=2, random_state=42))

    param_distribs = {
        'knnimputer__n_neighbors': randint(low=5, high=100),
        'adasyn__n_neighbors': randint(low=5, high=100),
        'svc__C': randint(low=1, high=100),
        'svc__coef0': randint(low=1, high=100)
    } 

    random_search =  RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distribs,
        n_iter=5,
        scoring="f1",
        verbose=4, 
        cv=5,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    if args.save_model:
        joblib.dump(random_search, 'parameters/svm_poly.pki')
    return random_search.predict(X_test)

def random_forest(X_train, y_train, X_test, args): 
    if args.best_parameters: 
        model = joblib.load('parameters/rf.pki')
        return model.predict(X_test)
    

    full_pipeline = make_pipeline(KNNImputer(), PolynomialFeatures(degree=2),
                                   ADASYN(), RobustScaler(), 
                                   RandomForestClassifier(max_depth=12, random_state=42))
    
    param_distibs = {
        'knnimputer__n_neighbors': randint(low=5, high=100),
        'adasyn__sampling_strategy': uniform(loc=0.1, scale=0.9),
        'adasyn__n_neighbors': randint(low=3, high=50),
        'randomforestclassifier__class_weight': [None, 'balanced'],
        'randomforestclassifier__criterion': ['gini', 'entropy'],
        'randomforestclassifier__n_estimators': randint(low=2, high=20),
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
    if args.save_model:
        joblib.dump(random_search, 'parameters/rf.pki')
    return random_search.predict(X_test)


def dense_network(X_train, y_train, X_test, args):
    
    pipeline = make_pipeline(KNNImputer(weights='distance', n_neighbors=100), PolynomialFeatures(degree=2), RobustScaler())
    X_train = pipeline.fit_transform(X_train, y_train)
    X_test = pipeline.transform(X_test)
    
    if args.best_parameters:
        model = tf.keras.models.load_model('parameters/dnn.h5')
        return [1 if p > 0.5 else 0 for p in model.predict(X_test)]

    y_train = tf.cast(y_train, dtype=tf.float32)

    adasyn = ADASYN(sampling_strategy='minority', n_neighbors=50)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

    X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_train_res, y_train_res, test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_res, y_train_res))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_res, y_val_res))
    val_dataset = val_dataset.batch(32)

    model = tf.keras.Sequential([
                tf.keras.layers.Dense(1000, activation="relu", input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(500, activation="relu"),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dense(25, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience = 5, mode='max', restore_best_weights = True)
    callbacks = [early_stopping_cb]

    
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=False,
        )
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.AUC(name='auc')])

    history = model.fit(train_dataset, epochs=70,
                        validation_data=val_dataset,
                        callbacks=callbacks)
    
    if args.save_model:
        model.save('parameters/dnn.h5')
    return [1 if p > 0.5 else 0 for p in model.predict(X_test)]
