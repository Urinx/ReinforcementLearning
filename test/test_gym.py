import gym

# env = gym.make('CartPole-v0')
# env = gym.make('Breakout-v0')
env = gym.make('Reacher-v2')
env.reset()

done = False
while not done:
    env.render()
    ob, reward, done, _ = env.step(env.action_space.sample())
env.close()