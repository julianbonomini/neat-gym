import gym
import neat
import os
import pickle
import multiprocessing
import numpy as np


runs_per_net = 5


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for _ in range(runs_per_net):
        env = gym.make('BipedalWalker-v3')
        observation = env.reset()

        # Run the given simulation for up until it fails.
        fitness = 0.0
        done = False
        counter = 0
        while not done:
          action = net.activate(observation)
          observation, reward, done, info = env.step(action)
          fitness += reward
          # Prevent forever loops, if no points in 250 ticks, then end the game
          if reward>0:
            counter = 0
          else:
            counter += 1
          if done or counter == 250:
                done = True

        fitnesses.append(fitness)


    # The genome's fitness is its mean performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    print('SAVING FILE!!!!!')
    # Save the winner.
    with open('walker/winner', 'wb') as f:
        pickle.dump(winner, f)

    print('SAVED. PRINTING WINNER: ')
    print(winner)


if __name__ == '__main__':
    run()