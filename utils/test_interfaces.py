import unittest
import numpy as np
import interfaces


class TestDataGenerator(unittest.TestCase):

    def test_Tape(self):

        tape_data = np.array([0, 1, 2, 3, 4, 5])
        InputTape = interfaces.Tape(tape_data, blank_char=6)

        self.assertEqual(InputTape.query(0), 0)

        InputTape.move_head(1)
        self.assertEqual(InputTape.read_head(), 1)
        InputTape.move_head(1)
        self.assertEqual(InputTape.read_head(), 2)
        InputTape.move_head(-1)
        self.assertEqual(InputTape.read_head(), 1)
        InputTape.move_head(0)
        self.assertEqual(InputTape.read_head(), 1)

        InputTape.write_head(5)
        self.assertEqual(InputTape.read_head(), 5)

        self.assertEqual(InputTape.query(-10), 6)


if __name__ == '__main__':
    unittest.main()
