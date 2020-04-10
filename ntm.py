import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime


class NTM():
    def __init__(self, memory_vector_len=10, num_char=5, max_run_time=200):
        self.mem_length = memory_vector_len
        self.mem_slots = 2
        self.num_char = num_char
        self.obs_length = 1
        self.init_model(memory_length=self.mem_length, num_char=self.num_char)
        self.init_memory()
        self.update_steps = 0
        self.max_run_time = max_run_time
        self.run_time = 0

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
            3, activation='softmax', name='input_head_control')(hidden_output)
        memory_head_control = tf.keras.layers.Dense(
            3, activation='softmax', name='memory_head_control')(hidden_output)
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

        self.last_output_pred = None
        self.output_pred_history = []
        self.output_target_history = []
        self.action_p_history = []

        log_dir = "logs/experiment/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.Writer = tf.summary.create_file_writer(log_dir)

        # Rl Params
        self.gamma = 0.99

        self.no_target_val = 0

    def init_memory(self):

        self.memory = tf.random.normal([self.mem_slots, self.mem_length])
        self.memory_head = 0

    def run_time_left(self, writes_left):

        if self.max_run_time - self.run_time <= writes_left:
            # shouldn't be negative
            assert((self.max_run_time - self.run_time) >= 0)
            return False
        else:
            return True

    def run(self, tape_input, writes_left):
        """Do Internal Computation inside the NTM"""

        tape_input = tf.constant(tape_input, dtype='float32')
        tape_input = tf.reshape(tape_input, [1, -1])
        memory_input = self.memory[self.memory_head, :]
        memory_input = tf.reshape(memory_input, [1, -1])

        with self.Tape as t:
            # [batch_size, vector_len]
            output = self.model(
                {'tape_input': tape_input, 'memory_input': memory_input})
            # output : [output_content, output_bool, input_head_control, memory_head_control, memory_content]
            output_pred = output[0]
            output_bool = output[1]
            input_head_control = output[2]
            memory_head_control = output[3]
            memory_content = output[4]

        # Sample Model Outputs
        sampled_output_pred = tf.random.categorical(
            tf.math.log(output_pred), num_samples=1).numpy()[0, 0]
        sampled_output_bool = tf.random.categorical(
            tf.math.log(output_bool), num_samples=1).numpy()[0, 0]
        sampled_input_head_control = tf.random.categorical(
            tf.math.log(input_head_control), num_samples=1).numpy()[0, 0]

        # Force write if run time left = writes_left
        if not self.run_time_left(writes_left):
            sampled_output_bool = 1

        with self.Tape as t:
            print(output_bool)
            p_action = output_bool[0, sampled_output_bool] * \
                input_head_control[0, sampled_input_head_control]

        self.last_output_pred = output_pred
        self.output_pred_history.append(output_pred)
        self.output_target_history.append(self.no_target_val)
        self.action_p_history.append(p_action)

        # over ride for now
        # sampled_output_bool = 1
        # sampled_input_head_control = 1

        self.run_time += 1

        # shift input head control by -1
        sampled_input_head_control -= 1
        return sampled_output_pred, sampled_output_bool, sampled_input_head_control

    def update_store(self, output_target, done):  # update only once done
        self.output_target_history[-1] = output_target
        if done:
            self.backup()

    def backup(self):

        num_actions = len(self.action_p_history)
        with self.Tape as tape:
            total_reward = tf.constant(0.)
            forward_returns = np.zeros([num_actions])

            for i in range(num_actions-1, -1, -1):
                output_target = self.output_target_history[i]
                output_pred_dist = self.output_pred_history[i]

                scalar_reward = 0
                if output_target != self.no_target_val:  # a step for which we made a prediction

                    # log probability of correct class
                    reward = tf.math.log(output_pred_dist[0, output_target])
                    total_reward += reward
                    scalar_reward = reward.numpy()

                # update forward looking return for policy gradient
                if i+1 == num_actions:
                    forward_returns[i] = scalar_reward
                else:
                    forward_returns[i] = scalar_reward + \
                        self.gamma*forward_returns[i+1]

            # print(forward_returns)

            # optimize this part to improve content
            obj_func = total_reward
            # optimize this part to improve log probability of good actions
            for j in range(num_actions):
                obj_func += tf.math.log(
                    self.action_p_history[j]) * forward_returns[j]

            obj_func *= -1  # flip around

        # print("Objective Function J(0): ", -1 * obj_func.numpy())
        with self.Writer.as_default():
            tf.summary.scalar('Objective Function', -1 * obj_func.numpy(),
                              step=self.update_steps)
        # update model parameters gradients
        gradients = self.Tape.gradient(
            obj_func, self.model.trainable_weights)

        self.Optimizer.apply_gradients([
            (grad, var)
            for (grad, var) in zip(gradients, self.model.trainable_variables)
            if grad is not None
        ])

        # Refresh Gradient Tape
        self.Tape = tf.GradientTape()

        # Clear run time
        self.run_time = 0

        # Clear Histories
        self.last_output_pred = None
        self.output_pred_history = []
        self.output_target_history = []
        self.action_p_history = []

        self.update_steps += 1
