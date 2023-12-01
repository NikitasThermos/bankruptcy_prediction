from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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