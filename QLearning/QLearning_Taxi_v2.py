import numpy as np
import gym
import random

env = gym.make("Taxi-v2")

action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))

# Hyperparameters
total_episodes = 50000
total_test_episodes = 100
max_steps = 99
learning_rate = 0.7
gamma = 0.618
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# Train
for episode in range(total_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        qtable[state, action] += learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        state = new_state
        if done: break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode+1))


# Play the Game
rewards = []
for episode in range(total_test_episodes):
    state = env.reset()
    total_rewards = 0

    print('='*20)
    print("[*] Episode", episode)
    print('='*20)

    for step in range(max_steps):
        env.render()
        action = np.argmax(qtable[state, :])
        state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            print('[*] Score', total_rewards)
            break

env.close()
print('[*] Average Score: ' + str(sum(rewards) / total_test_episodes))