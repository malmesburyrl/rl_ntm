import numpy as np
import sys
sys.path.insert(0, 'utils')

import data_generator  # noqa
import interfaces  # noqa


class Copy():
    def __init__(self, string_len=5, max_char=5):
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

        self.target = self.TargetTape.tape_data

        self.curr_step = 0
        self.num_writes = 0

    def writes_remaining(self):
        return self.string_len - self.num_writes

    def reset(self):
        self.__init__(self.string_len, self.max_char)
        observation = self.InputTape.read_head()
        target = self.TargetTape.read_head()  # expected output
        done = False
        info = self.writes_remaining()

        return observation, target, done, info

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
            self.num_writes += 1

            self.OutputTape.write_head(output_content)

            self.OutputTape.move_head(1)
            self.TargetTape.move_head(1)

        done = False
        info = self.writes_remaining()
        if self.writes_remaining() == 0:
            done = True
        assert(self.writes_remaining() >= 0)  # should not go below 0
        return observation, target, done, info


if __name__ == '__main__':

    env = Copy()
    print(env.InputTape.tape_data)
    observation, target, done, info = env.step((0, 0, 0))
    print(observation, target, done, info)
    env.render()

    observation, target, done, info = env.step((0, 1, 1))
    print(observation, target, done, info)
    env.render()

    observation, target, done, info = env.step((0, 1, 1))
    print(observation, target, done, info)
    env.render()

    observation, target, done, info = env.step((0, 1, 1))
    print(observation, target, done, info)
    env.render()

    observation, target, done, info = env.step((0, 1, 1))
    print(observation, target, done, info)
    env.render()

    observation, target, done, info = env.step((0, 1, 1))
    print(observation, target, done, info)
    env.render()

    observation, target, done, info = env.step((0, 1, 1))
    print(observation, target, done, info)
    env.render()
