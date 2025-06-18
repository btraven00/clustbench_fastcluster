#!/usr/bin/env python

"""
Unit tests for the set_seed context manager in prng.py
"""

import unittest
import numpy as np
import random
import sys
from pathlib import Path

# Add parent directory to path to import run_genieclust
sys.path.insert(0, str(Path(__file__).parent))
from prng import set_seed

class TestSetSeedContext(unittest.TestCase):
    def test_reproducibility_numpy(self):
        """Test that numpy random values are reproducible with the same seed"""
        # Generate values with seed 42
        with set_seed(42):
            values1 = np.random.rand(10)

        # Generate values again with the same seed
        with set_seed(42):
            values2 = np.random.rand(10)

        # Values should be identical
        np.testing.assert_array_equal(values1, values2)

        # Generate values with a different seed
        with set_seed(43):
            values3 = np.random.rand(10)

        # Values should be different
        self.assertFalse(np.array_equal(values1, values3),
                         "Results with different seeds should be different")

    def test_reproducibility_python(self):
        """Test that Python's random values are reproducible with the same seed"""
        # Generate values with seed 42
        with set_seed(42):
            values1 = [random.random() for _ in range(10)]

        # Generate values again with the same seed
        with set_seed(42):
            values2 = [random.random() for _ in range(10)]

        # Values should be identical
        self.assertEqual(values1, values2)

        # Generate values with a different seed
        with set_seed(43):
            values3 = [random.random() for _ in range(10)]

        # Values should be different
        self.assertNotEqual(values1, values3,
                           "Results with different seeds should be different")

    def test_mixed_reproducibility(self):
        """Test that mixed numpy and Python random values are reproducible"""
        # Generate values with seed 42
        with set_seed(42):
            np_values1 = np.random.rand(10)
            py_values1 = [random.random() for _ in range(10)]

        # Generate values again with the same seed
        with set_seed(42):
            np_values2 = np.random.rand(10)
            py_values2 = [random.random() for _ in range(10)]

        # Both numpy and Python values should be identical
        np.testing.assert_array_equal(np_values1, np_values2)
        self.assertEqual(py_values1, py_values2)

    def test_state_restoration(self):
        """Test that the random state is properly restored after context exit"""
        # Set a known seed
        np.random.seed(100)
        random.seed(100)

        # Get initial values
        initial_np = np.random.rand(5)
        initial_py = [random.random() for _ in range(5)]

        # Use set_seed with a different seed
        with set_seed(42):
            # Values inside the context should be different
            context_np = np.random.rand(5)
            context_py = [random.random() for _ in range(5)]

            # These should be different from initial values
            self.assertFalse(np.array_equal(initial_np, context_np))
            self.assertNotEqual(initial_py, context_py)

        # After context exit, state should be restored
        after_np = np.random.rand(5)
        after_py = [random.random() for _ in range(5)]

        # Reset to the same initial state to get expected values
        np.random.seed(100)
        random.seed(100)

        # Skip the initial values we already generated
        _ = np.random.rand(5)
        _ = [random.random() for _ in range(5)]

        # These are the values we should get after context
        expected_np = np.random.rand(5)
        expected_py = [random.random() for _ in range(5)]

        # Verify state was properly restored
        np.testing.assert_array_equal(after_np, expected_np)
        self.assertEqual(after_py, expected_py)

    def test_nested_contexts(self):
        """Test that nested set_seed contexts work correctly"""
        with set_seed(100):
            outer_values = np.random.rand(5)

            with set_seed(200):
                inner_values = np.random.rand(5)

                # Inner context should have different values
                self.assertFalse(np.array_equal(outer_values, inner_values))

            # After inner context, should return to outer context's state
            after_inner = np.random.rand(5)

            # Generate expected values by resetting to the same seed
            with set_seed(100):
                _ = np.random.rand(5)  # Skip the first batch
                expected_after_inner = np.random.rand(5)

            np.testing.assert_array_equal(after_inner, expected_after_inner)

    def test_none_seed(self):
        """Test that set_seed with None doesn't change the state"""
        # Set a known seed
        np.random.seed(100)
        random.seed(100)

        # Get first batch of values
        values1 = np.random.rand(5)

        # Use set_seed with None seed
        with set_seed(None):
            # Should continue the same sequence
            values2 = np.random.rand(5)

        # After context
        values3 = np.random.rand(5)

        # Reset to check expected sequence
        np.random.seed(100)
        random.seed(100)

        expected1 = np.random.rand(5)
        expected2 = np.random.rand(5)
        expected3 = np.random.rand(5)

        # Verify all values follow the expected sequence
        np.testing.assert_array_equal(values1, expected1)
        np.testing.assert_array_equal(values2, expected2)
        np.testing.assert_array_equal(values3, expected3)


if __name__ == "__main__":
    unittest.main()
