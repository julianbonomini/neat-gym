import gym
import numpy as np
np.set_printoptions(threshold=np.inf)

env = gym.make('BipedalWalker-v3')
observation = env.reset()
print(env.observation_space)
print(env.action_space)

done = False
while not done:
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  env.render()