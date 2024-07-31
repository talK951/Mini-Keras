import unittest
import numpy as np
from ErrorFunctions import MSE, MAE, RMSE, LogLossError, ErrorFunction


class TestErrorFunctions(unittest.TestCase):

    def test_error_function_relationships(self):
        assert issubclass(MSE, ErrorFunction)
        assert issubclass(MAE, ErrorFunction)
        assert issubclass(RMSE, ErrorFunction)
        assert issubclass(LogLossError, ErrorFunction)

    def test_MSE_function(self):
        error_fn = MSE()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Simple case with positive numbers"

        # Test Case 2: Simple case with a mix of positive and negative numbers
        y_true = np.array([1, -2, 3])
        y_pred = np.array([1, -2, 3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Simple case with a mix of positive and negative numbers"

        # Test Case 3: Case with different predictions
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 4])
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert error_fn(y_true, y_pred) == expected_mse, "Failed: Case with different predictions"

        # Test Case 4: Case with zeros
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with zeros"

        # Test Case 5: Case with one element
        y_true = np.array([1])
        y_pred = np.array([1])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with one element"

        # Test Case 6: Case with large numbers
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1e10, 2e10, 3e10])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with large numbers"

        # Test Case 7: Case with negative numbers
        y_true = np.array([-1, -2, -3])
        y_pred = np.array([-1, -2, -3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with negative numbers"

        # Test Case 8: Case with decimal numbers
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with decimal numbers"

        # Test Case 9: Edge case with empty arrays
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(Exception):
            error_fn(y_true, y_pred)

    def test_MAE_function(self):
        error_fn = MAE()

        # Test Case 1: Simple case with positive numbers
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Simple case with positive numbers"

        # Test Case 2: Simple case with a mix of positive and negative numbers
        y_true = np.array([1, -2, 3])
        y_pred = np.array([1, -2, 3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Simple case with a mix of positive and negative numbers"

        # Test Case 3: Case with different predictions
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 4])
        expected_mae = np.mean(np.abs(y_true - y_pred))
        assert error_fn(y_true, y_pred) == expected_mae, "Failed: Case with different predictions"

        # Test Case 4: Case with zeros
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with zeros"

        # Test Case 5: Case with one element
        y_true = np.array([1])
        y_pred = np.array([1])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with one element"

        # Test Case 6: Case with large numbers
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1e10, 2e10, 3e10])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with large numbers"

        # Test Case 7: Case with negative numbers
        y_true = np.array([-1, -2, -3])
        y_pred = np.array([-1, -2, -3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with negative numbers"

        # Test Case 8: Case with decimal numbers
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with decimal numbers"

        # Test Case 9: Edge case with empty arrays
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(Exception):
            error_fn(y_true, y_pred)

    def test_RMSE_function(self):
        error_fn = RMSE()
        # Test Case 1: Simple case with positive numbers
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Simple case with positive numbers"

        # Test Case 2: Simple case with a mix of positive and negative numbers
        y_true = np.array([1, -2, 3])
        y_pred = np.array([1, -2, 3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Simple case with a mix of positive and negative numbers"

        # Test Case 3: Case with different predictions
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 4])
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert error_fn(y_true, y_pred) == expected_rmse, "Failed: Case with different predictions"

        # Test Case 4: Case with zeros
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with zeros"

        # Test Case 5: Case with one element
        y_true = np.array([1])
        y_pred = np.array([1])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with one element"

        # Test Case 6: Case with large numbers
        y_true = np.array([1e10, 2e10, 3e10])
        y_pred = np.array([1e10, 2e10, 3e10])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with large numbers"

        # Test Case 7: Case with negative numbers
        y_true = np.array([-1, -2, -3])
        y_pred = np.array([-1, -2, -3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with negative numbers"

        # Test Case 8: Case with decimal numbers
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with decimal numbers"

        # Test Case 9: Edge case with empty arrays
        y_true = np.array([])
        y_pred = np.array([])
        with self.assertRaises(Exception):
            error_fn(y_true, y_pred)

    def test_log_loss_error_function(self):
        error_fn = LogLossError()
        # Test Case 1: Perfect predictions
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        assert error_fn(y_true, y_pred) == 0, "Failed: Perfect predictions"

        # Test Case 2: Completely wrong predictions
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        assert error_fn(y_true, y_pred) == 0, "Failed: Completely wrong predictions"

        # Test Case 3: Mixed predictions
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        assert error_fn(y_true, y_pred) == 0, "Failed: Mixed predictions"

        # Test Case 4: Predictions with values close to 0 and 1
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1e-10, 1 - 1e-10, 1 - 1e-10, 1e-10])
        assert error_fn(y_true, y_pred) == 0, "Failed: Predictions with values close to 0 and 1"

        # Test Case 5: Case with one element
        y_true = np.array([1])
        y_pred = np.array([1])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with one element"

        # Test Case 6: Case with one wrong element
        y_true = np.array([0])
        y_pred = np.array([1])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with one wrong element"

        # Test Case 7: Case with decimal probabilities
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.7, 0.3, 0.9, 0.4])
        assert error_fn(y_true, y_pred) == 0, "Failed: Case with decimal probabilities"

        # Test Case 8: Edge case with empty arrays
        y_true = np.array([])
        y_pred = np.array([])
        assert error_fn(y_true, y_pred) == 0
