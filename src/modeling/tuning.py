def tune_hyperparameters(model, param_grid, X_train, y_train, scoring='accuracy', cv=5):
    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def random_search_hyperparameters(model, param_distributions, X_train, y_train, scoring='accuracy', n_iter=100, cv=5):
    from sklearn.model_selection import RandomizedSearchCV

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, scoring=scoring, cv=cv)
    random_search.fit(X_train, y_train)
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def tune_model(model, X_train, y_train, tuning_method='grid', param_grid=None, param_distributions=None, scoring='accuracy', cv=5, n_iter=100):
    if tuning_method == 'grid':
        if param_grid is None:
            raise ValueError("param_grid must be provided for grid search.")
        return tune_hyperparameters(model, param_grid, X_train, y_train, scoring, cv)
    elif tuning_method == 'random':
        if param_distributions is None:
            raise ValueError("param_distributions must be provided for random search.")
        return random_search_hyperparameters(model, param_distributions, X_train, y_train, scoring, n_iter, cv)
    else:
        raise ValueError("Invalid tuning method. Choose 'grid' or 'random'.")