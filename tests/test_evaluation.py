import unittest
from src.evaluation.metrics import accuracy, precision, recall, f1_score
from src.evaluation.report import generate_report

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        self.y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    def test_accuracy(self):
        result = accuracy(self.y_true, self.y_pred)
        expected = 0.8
        self.assertAlmostEqual(result, expected)

    def test_precision(self):
        result = precision(self.y_true, self.y_pred)
        expected = 0.75
        self.assertAlmostEqual(result, expected)

    def test_recall(self):
        result = recall(self.y_true, self.y_pred)
        expected = 0.75
        self.assertAlmostEqual(result, expected)

    def test_f1_score(self):
        result = f1_score(self.y_true, self.y_pred)
        expected = 0.75
        self.assertAlmostEqual(result, expected)

    def test_generate_report(self):
        report = generate_report(self.y_true, self.y_pred)
        self.assertIn('Accuracy', report)
        self.assertIn('Precision', report)
        self.assertIn('Recall', report)
        self.assertIn('F1 Score', report)

if __name__ == '__main__':
    unittest.main()