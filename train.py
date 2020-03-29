import sys
sys.path.insert(0, 'utils')  # noqa

import tasks
import ntm
import interfaces


# Training Params
train_iter = 10
batch_size = 10

# Task Params
max_epsiode_length = 100

# Define Rewards
correct_output_reward = 1
no_action_reward = 0
incorrect_output_reward = -1

Task = tasks.Copy()

for i in range(train_iter):
    pass
