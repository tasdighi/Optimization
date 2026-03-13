import unittest
import numpy as np
import pandas as pd
from pipelines import batch_analysis
from strategies import NSGA3Strategy

class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        days = np.repeat(range(5), 24)  # 5 days, 24 hours each
        data_values = np.random.randn(120)  # 120 data points
        self.df = pd.DataFrame({'day': days, 'data_values': data_values})

    def test_batch_analysis(self):
        strategy = NSGA3Strategy(pop_size=50)
        result = batch_analysis(self.df, strategy, 0, 2, 1000.0)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Assuming positive metric

if __name__ == '__main__':
    unittest.main()