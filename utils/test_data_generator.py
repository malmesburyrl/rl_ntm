import unittest
import numpy as np
import data_generator


class TestDataGenerator(unittest.TestCase):

    def test_random_tape(self):

        tape = data_generator.random_tape(10, 5)

        self.assertEqual(len(tape), 10)
        self.assertTrue(np.amax(tape) <= 5)


if __name__ == '__main__':
    unittest.main()
