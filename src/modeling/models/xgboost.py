from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class XGBoostModel:
    def __init__(self, params=None):
        self.model = XGBClassifier(**(params if params else {}))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy