import numpy as np
import sys
sys.path.insert(0, 'utils')

import data_generator  # noqa
import interfaces  # noqa


class Copy():
    def __init__(self, string_len=5, max_char=5, max_step=100):
        self.string_len = string_len
        self.max_char = max_char
        self.blank_char = max_char+1
        assert(self.blank_char <= 9)

        tape_data = data_generator.random_tape(
            string_len, max_char=self.max_char)
        target_data = tape_data
        output_tape = np.ones_like(
            tape_data)*self.blank_char  # tape of empty char

        self.InputTape = interfaces.Tape(tape_data, blank_char=self.blank_char)
        self.OutputTape = interfaces.Tape(
            output_tape, blank_char=self.blank_char)
        self.TargetTape = interfaces.Tape(
            target_data, blank_char=self.blank_char)

        self.max_step = max_step
        self.curr_step = 0

        # Define Rewards
        self.correct_output_reward = 1
        self.no_action_reward = 0
        self.incorrect_output_reward = -1

    def reset(self):
        self.__init__(self.string_len, self.max_char, self.max_step)
        observation = self.InputTape.read_head()
        reward = 0
        target = self.TargetTape.read_head()  # expected output
        done = False
        info = None

        return observation, reward, target, done, info

    def render(self):
        print("Input Tape: ", self.InputTape.display())
        print("Output Tape: ", self.OutputTape.display())
        print("Target Tape: ", self.TargetTape.display())

    def step(self, action):

        input_head_action = action[0]
        output_write_action = action[1]
        output_content = action[2]

        # input interface
        self.InputTape.move_head(input_head_action)
        observation = self.InputTape.read_head()

        # output interface
        target = self.TargetTape.read_head()  # expected output
        if output_write_action == 1:
            self.OutputTape.write_head(output_content)

            self.OutputTape.move_head(1)
            self.TargetTape.move_head(1)

        reward = 0
        done = False
        info = None

        self.curr_step += 1
        if self.curr_step > self.max_step:
            done = True
        return observation, reward, target, done, info


if __name__ == '__main__':

    env = Copy()
    print(env.InputTape.tape_data)
    env.step((0, 0, 0))
    env.render()

    env.step((0, 1, 1))
    env.render()
