import numpy as np
import gym
import random

env = gym.make('FrozenLake-v0')
action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

total_episodes = 1000
learning_rate = 0.8
max_steps = 99
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

rewards = []
for episode in range(total_episodes):
    state = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state
        total_rewards += reward
        if done: break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode+1))
    rewards.append(total_rewards)

    print('[*] episode {}, total reward {}, average score {}'.format(episode, total_rewards, sum(rewards)/(episode+1)))

print(qtable)

# Play the game

for episode in range(1):
    state = env.reset()
    print('*'*20)
    print('EPISODE ', episode)

    for step in range(max_steps):
        env.render()
        action = np.argmax(qtable[state])
        input()
        state, reward, done, info = env.step(action)
        if done: break

env.close()

