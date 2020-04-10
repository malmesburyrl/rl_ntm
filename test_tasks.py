import unittest
import numpy as np
import tasks


class TestDataGenerator(unittest.TestCase):

    def test_Copy(self):

        max_char = 3
        string_len = 3
        env = tasks.Copy(string_len=string_len, max_char=max_char)
        # actions
        # input_head_action = action[0]
        # output_write_action = action[1]
        # output_content = action[2]
        blank_char = max_char + 1
        observation, target, done, info = env.reset()
        full_target = env.target
        print(full_target)

        observation, target, done, info = env.step((-1, 0, 0))
        self.assertTrue(observation == blank_char)

        observation, target, done, info = env.step((1, 1, 0))
        self.assertTrue(observation == full_target[0])
        self.assertTrue(info == string_len - 1)
        self.assertFalse(done)

        observation, target, done, info = env.step((1, 1, 0))
        self.assertTrue(observation == full_target[1])
        self.assertTrue(info == string_len - 2)
        self.assertFalse(done)
        env.render()

        observation, target, done, info = env.step((1, 1, 0))
        self.assertTrue(observation == full_target[2])
        self.assertTrue(info == string_len - 3)
        self.assertTrue(done)
        env.render()


if __name__ == '__main__':
    unittest.main()
