import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class NTM():
    def __init__(self, memory_vector_len=10, num_char=5):
        self.mem_length = memory_vector_len
        self.mem_slots = 2
        self.num_char = num_char
        self.obs_length = 1
        self.init_model(memory_length=self.mem_length, num_char=self.num_char)
        self.init_memory()
        self.update_steps = 0

    def init_model(self, memory_length, num_char):

        input_tape_input = tf.keras.layers.Input(
            shape=(1,), dtype='float32', name='tape_input')
        memory_input = tf.keras.layers.Input(
            shape=(memory_length,), dtype='float32', name='memory_input')
        input = tf.keras.layers.concatenate(
            [input_tape_input, memory_input], axis=1)

        l1 = tf.keras.layers.Dense(32, activation='relu')(input)
        l2 = tf.keras.layers.Dense(32, activation='relu')(l1)

        hidden_output = l2

        output_content = tf.keras.layers.Dense(
            num_char, activation='softmax', name='output_content')(hidden_output)
        output_bool = tf.keras.layers.Dense(
            2, activation='softmax', name='output_bool')(hidden_output)
        input_head_control = tf.keras.layers.Dense(
            2, activation='softmax', name='input_head_control')(hidden_output)
        memory_head_control = tf.keras.layers.Dense(
            2, activation='softmax', name='memory_head_control')(hidden_output)
        memory_content = tf.keras.layers.Dense(
            memory_length, name='memory_content')(hidden_output)

        # self.model = tf.keras.models.Sequential()
        # self.model.add(tf.keras.layers.Dense(128, activation='relu',
        #                                      input_shape=(input_size,)))
        # # model.add(Dropout(0.5))
        # self.model.add(tf.keras.layers.Dense(
        #     10, activation='softmax'))

        self.model = tf.keras.models.Model(
            inputs=[input_tape_input, memory_input],
            outputs=[output_content, output_bool, input_head_control, memory_head_control, memory_content])
        self.model.summary()

        self.Tape = tf.GradientTape()
        self.Optimizer = keras.optimizers.SGD()

    def init_memory(self):

        self.memory = tf.random.normal([self.mem_slots, self.mem_length])
        self.memory_head = 0

    def input(self, inputs):
        self.input_content = np.array(inputs[0]).reshape(1, 1)
        self.memory_content = inputs[1]
        self.input = np.vstack(self.input_content, self.memory_content)
        assert(self.input_content.shape == (1, 1))
        assert(self.memory_content.shape == (self.mem_length, 1))

        return None

    def run(self, tape_input, output_target):
        """Do Internal Computation inside the NTM"""

        tape_input = tf.constant(tape_input, dtype='float32')
        tape_input = tf.reshape(tape_input, [1, -1])
        memory_input = self.memory[self.memory_head, :]
        memory_input = tf.reshape(memory_input, [1, -1])

        output_target = tf.one_hot(output_target, self.num_char)

        with self.Tape as t:
            # [batch_size, vector_len]
            output = self.model(
                {'tape_input': tape_input, 'memory_input': memory_input})
            #output : [output_content, output_bool, input_head_control, memory_head_control, memory_content]
            output_pred = output[0]
            output_bool = output[1]
            input_head_control = output[2]
            memory_head_control = output[3]
            memory_content = output[4]

            if True:
                output_loss = keras.losses.categorical_crossentropy(
                    output_target, output_pred)

        if True:
            # if self.update_steps % 10 == 0:
            print("Loss: ", output_loss.numpy())
            self.update(output_loss)

        # over ride for now
        output_bool = tf.constant([[0., 1.]])
        input_head_control = tf.constant([[0., 1.]])
        print(output_bool.numpy().shape)
        # print(output_bool)
        sampled_output_pred = tf.random.categorical(
            tf.math.log(output_pred), num_samples=1)
        sampled_output_bool = tf.random.categorical(
            tf.math.log(output_bool), num_samples=1)
        sampled_input_head_control = tf.random.categorical(
            tf.math.log(input_head_control), num_samples=1)

        return sampled_output_pred.numpy()[0, 0], sampled_output_bool.numpy()[0, 0], sampled_input_head_control.numpy()[0, 0]

    def update(self, fx):
        grads = self.Tape.gradient(fx, self.model.trainable_weights)
        # self.Optimizer.apply_gradients(
        #     zip(grads, self.model.trainable_weights))

        # Hacky way so you don't get errors for variables which aren't training
        self.Optimizer.apply_gradients([
            (grad, var)
            for (grad, var) in zip(grads, self.model.trainable_variables)
            if grad is not None
        ])

        self.Tape = tf.GradientTape()

        self.update_steps += 1
        # print(grads)

    def random_output(self):

        input_control = np.random.randint(2)  # [0,1]
        memory_control = 0  # [0,1]
        # vector len(M) where M is the size of memory vectors
        memory_content = np.zeros(10)
        output_control = np.random.randint(2)  # [0,1]
        # vector len(V) where V is num of characters in vocabulary
        output_content = np.random.randint(5)

        return (input_control, memory_control, memory_content, output_control, output_content)
