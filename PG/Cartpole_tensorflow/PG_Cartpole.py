import tensorflow as tf
import numpy as np
import gym
import sys
import time


def create_environment():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(1)

    state = env.reset()
    state_size = len(state)
    action_size = env.action_space.n

    return env, state_size, action_size

def test_environment():
    env, _, _ = create_environment()
    episodes = 1

    for _ in range(episodes):
        print(env.reset())
        env.render()
        total_rewards = 0
        done = False

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            env.render()

            total_rewards += reward
            print('action:', action, 'reward:', reward)
            time.sleep(0.5)

        print('[*] Total Reward:',total_rewards)

def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std

    return discounted_episode_rewards


class PGNetwork():

    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.name_scope(name):
            self.input_state = tf.placeholder(tf.float32, [None, state_size], name='input_state')
            self.input_action = tf.placeholder(tf.int32, [None, action_size], name='input_action')
            self.input_rewards = tf.placeholder(tf.float32, [None, ], name='input_rewards')
            self.input_mean_reward = tf.placeholder(tf.float32, name='input_mean_reward')

            fc1 = tf.contrib.layers.fully_connected(
                inputs = self.input_state,
                num_outputs = 10,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer())
            fc2 = tf.contrib.layers.fully_connected(
                inputs = fc1,
                num_outputs = action_size,
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.layers.xavier_initializer())
            fc3 = tf.contrib.layers.fully_connected(
                inputs = fc2,
                num_outputs = action_size,
                activation_fn = None,
                weights_initializer = tf.contrib.layers.xavier_initializer())

            self.output_action = tf.nn.softmax(fc3)
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.input_action)
            self.loss = tf.reduce_mean(neg_log_prob * self.input_rewards)
            self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



def train():
    env, state_size, action_size = create_environment()
    # Hyperparameters
    max_episodes = 10000
    learning_rate = 0.01
    gamma = 0.95

    tf.reset_default_graph()
    PG = PGNetwork(state_size, action_size, learning_rate)

    writer = tf.summary.FileWriter('PG_Cartpole_log')
    tf.summary.scalar('Loss', PG.loss)
    tf.summary.scalar('Reward mean', PG.input_mean_reward)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()


    all_rewards = []
    total_rewards = 0
    maximum_reward_recorded = 0
    episode_states, episode_actions, episode_rewards = [], [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(max_episodes):
            episode_rewards_sum = 0
            state = env.reset()
            env.render()
            done = False

            while not done:
                output_action = sess.run(PG.output_action, feed_dict={PG.input_state: state.reshape([1, 4])})
                action = np.random.choice(range(action_size), p=output_action.ravel())

                new_state, reward, done, info = env.step(action)
                env.render()
                
                episode_states.append(state)
                a = np.zeros(action_size)
                a[action] = 1
                episode_actions.append(a)
                episode_rewards.append(reward)

                state = new_state

            episode_rewards_sum = np.sum(episode_rewards)
            all_rewards.append(episode_rewards_sum)
            total_rewards = np.sum(all_rewards)
            mean_reward = np.divide(total_rewards, episode + 1)
            maximum_reward_recorded = np.amax(all_rewards)

            print('='*20)
            print('Episode:', episode)
            print('Reward:', episode_rewards_sum)
            print('Mean Reward:', mean_reward)
            print('Max reward so far:', maximum_reward_recorded)

            episode_rewards = discount_and_normalize_rewards(episode_rewards, gamma)
            loss, _ = sess.run([PG.loss, PG.train], feed_dict={
                    PG.input_state: np.vstack(np.array(episode_states)),
                    PG.input_action: np.vstack(np.array(episode_actions)),
                    PG.input_rewards: episode_rewards
                })

            summary = sess.run(write_op, feed_dict={
                    PG.input_state: np.vstack(np.array(episode_states)),
                    PG.input_action: np.vstack(np.array(episode_actions)),
                    PG.input_rewards: episode_rewards,
                    PG.input_mean_reward: mean_reward
                })

            writer.add_summary(summary, episode)
            writer.flush()
            episode_states, episode_actions, episode_rewards = [], [], []

            if episode % 5 == 0:
                save_path = saver.save(sess, './model/model.ckpt')
                print('[*] Model Saved:', save_path)

        print('Train done')

def play():
    env, state_size, action_size = create_environment()
    learning_rate = 0.01

    with tf.Session() as sess:
        PG = PGNetwork(state_size, action_size, learning_rate)
        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")

        state = env.reset()
        env.render()
        done = False
        episode_rewards = []
        while not done:
                output_action = sess.run(PG.output_action, feed_dict={PG.input_state: state.reshape([1, 4])})
                action = np.random.choice(range(action_size), p=output_action.ravel())

                state, reward, done, info = env.step(action)
                env.render()
                episode_rewards.append(reward)
        
        episode_rewards_sum = np.sum(episode_rewards)
        print('Episode Rewards:', episode_rewards_sum)

if __name__ == '__main__':
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play()

















