class DecisionTreeModel:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return self.model.score(X, y)