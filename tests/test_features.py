import unittest
from src.features.selection import select_features
from src.features.generation import generate_features

class TestFeatureFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        }

    def test_select_features(self):
        selected_features = select_features(self.data, target='target')
        # Assuming select_features returns a list of selected feature names
        self.assertIn('feature1', selected_features)
        self.assertIn('feature2', selected_features)

    def test_generate_features(self):
        generated_data = generate_features(self.data)
        # Assuming generate_features adds a new feature 'feature1_feature2'
        self.assertIn('feature1_feature2', generated_data.columns)

if __name__ == '__main__':
    unittest.main()