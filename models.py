from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
tf.random.set_seed(42)

def sgd(X_train, y_train, X_val):
    sgd_clf = SGDClassifier(random_state=42)
    param_grid = {'alpha':[0.0003, 0.0005, 0.0007], 'tol':[1e-4, 1e-3, 1e-2],
                  'epsilon': [0.05, 0.1, 0.2]}
    grid_search = GridSearchCV(sgd_clf, param_grid, cv=5, 
                               scoring='accuracy',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    print('SGD best parameters:')
    print(grid_search.best_params_)

    return grid_search.predict(X_val)

def random_forest(X_train, y_train, X_val): 
    forest_clf = RandomForestClassifier(random_state=42, criterion='entropy')
    param_grid = {'n_estimators':[4, 5, 6], 'min_samples_split':[6, 8, 10, 12]}
    grid_search = GridSearchCV(forest_clf, param_grid, 
                                cv=3, scoring='accuracy',
                                return_train_score=True, verbose=True)
    grid_search.fit(X_train, y_train)
    print('Random Forest best parameters:')
    print(grid_search.best_params_)
    return grid_search.predict(X_val)

def dense_network(X_train, y_train, X_val, y_val):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_dataset = train_dataset.shuffle(buffer_size=3000).batch(32)
    val_dataset = val_dataset.batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                         restore_best_weights=True)
    callabacks = [early_stopping_cb]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                         beta_1=0.9, beta_2=0.999)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    history = model.fit(train_dataset, epochs=30,
                        validation_data=val_dataset,
                        callabacks=callabacks)

    return model.predict(X_val)