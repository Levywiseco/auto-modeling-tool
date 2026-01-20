import unittest
from src.modeling.train import train_model
from src.modeling.models.logistic import LogisticModel
from src.modeling.models.tree import TreeModel
from src.modeling.models.xgboost import XGBoostModel

class TestModeling(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize data for testing
        self.X_train = [[0, 0], [1, 1]]
        self.y_train = [0, 1]
        self.X_test = [[0, 1], [1, 0]]
        self.y_test = [1, 0]

    def test_logistic_model(self):
        model = LogisticModel()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_tree_model(self):
        model = TreeModel()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_xgboost_model(self):
        model = XGBoostModel()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

if __name__ == '__main__':
    unittest.main()