import numpy as np
import tensorflow as tf


class NTM():
    pass

    def input(self, inputs):
        input_content = inputs[0]
        memory_content = inputs[1]
        return None

    def output(self):

        input_control = 0  # [-1,0,1]
        memory_control = 0  # [-1,0,1]
        # vector len(M) where M is the size of memory vectors
        memory_content = np.zeros(10)
        output_control = 0  # [0,1]
        # vector len(V) where V is num of characters in vocabulary
        output_content = np.zeros(10)

        return (input_control, memory_control, memory_content, output_control, output_content)
