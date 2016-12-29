"""
### NOTICE ###

You DO NOT need to upload this file.

"""
import sys, random
import tensorflow as tf

from agent import Agent
from environment import ALE

tf.set_random_seed(123)
random.seed(123)

init_seed = int(sys.argv[1])
init_rand = int(sys.argv[2])

with tf.Session() as sess:

    # Init env
    env = ALE(init_seed, init_rand)

    # Init agent
    agent = Agent(sess, env.ale.getMinimalActionSet())
    action_repeat, screen_type = agent.getSetting()

    # Set env setting
    env.setSetting(action_repeat, screen_type)

    # Get a new game
    #screen = env.new_game()
    
    # Start playing
    current_reward = 0
    total_reward = 0
    for episode in range(100000000):
        screen = env.new_game()
        current_reward = 0
        while True:
            action = agent.play(screen)
            reward, screen, terminal = env.act(action)
            current_reward += reward
            if terminal:
                break
        total_reward += current_reward
        print("%d %d %d" % (episode, current_reward, total_reward / (episode + 1)))

    print("%d %d" % (seed, current_reward))
