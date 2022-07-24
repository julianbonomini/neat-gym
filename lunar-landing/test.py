import os
import pickle
import neat
import gym
import numpy as np

with open('lunar-landing/winner', 'rb') as f:
  c = pickle.load(f)

print('Loaded genome', c)


# Load config file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

neat = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('LunarLander-v2')
observation = env.reset()

done = False
while not done:
  action = np.argmax(neat.activate(observation))

  observation, reward, done, info = env.step(action)
  env.render()