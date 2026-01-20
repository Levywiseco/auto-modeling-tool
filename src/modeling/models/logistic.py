class LogisticRegressionModel:
    def __init__(self, solver='lbfgs', max_iter=100):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(solver=solver, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)