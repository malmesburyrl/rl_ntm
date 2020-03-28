import numpy as np
# import tensorflow as tf
import gym


env = gym.make('Copy-v0')
print(env.action_space)
print(env.observation_space)

# print(env.observation_space.high)
# print(env.observation_space.low)

observation = env.reset()
env.render()
observation, reward, done, info = env.step(action)
print()
#
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
