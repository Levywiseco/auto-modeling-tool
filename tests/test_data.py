import unittest
from src.data.loaders import load_data
from src.data.preprocess import preprocess_data

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.test_data_path = 'path/to/test/data.csv'
        self.data = load_data(self.test_data_path)

    def test_data_loading(self):
        self.assertIsNotNone(self.data)
        self.assertGreater(len(self.data), 0)

    def test_data_preprocessing(self):
        processed_data = preprocess_data(self.data)
        self.assertIsNotNone(processed_data)
        self.assertGreater(len(processed_data), 0)
        self.assertTrue(all(processed_data.notnull().all()))

if __name__ == '__main__':
    unittest.main()