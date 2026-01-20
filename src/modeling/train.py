def train_model(X_train, y_train, model_type='logistic', **kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    if model_type == 'logistic':
        model = LogisticRegression(**kwargs)
    elif model_type == 'tree':
        model = DecisionTreeClassifier(**kwargs)
    elif model_type == 'xgboost':
        model = XGBClassifier(**kwargs)
    else:
        raise ValueError("Unsupported model type. Choose from 'logistic', 'tree', or 'xgboost'.")

    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)