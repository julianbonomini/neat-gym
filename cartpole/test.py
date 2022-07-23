import os
import pickle
import neat
import gym
import numpy as np

with open('cartpole/winner', 'rb') as f:
  c = pickle.load(f)

print('Loaded genome', c)


# Load config file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

neat = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('CartPole-v1', render_mode='human')
observation = env.reset()

done = False
for _ in range(300):
  action = np.argmax(neat.activate(observation))

  observation, reward, done, info = env.step(action)
  env.render(mode='rgb-array')