import tensorflow as tf
import numpy as np
from vizdoom import DoomGame
import random
import time
from skimage import transform
from collections import deque
import sys

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


###########################################
# Constant
stack_size = 4
frame_size = (100, 160)
# Global variables
stacked_frames = deque([np.zeros(frame_size) for _ in range(stack_size)], maxlen=stack_size)
###########################################


def create_environment():
    game = DoomGame()
    game.load_config('defend_the_center.cfg')
    game.set_doom_scenario_path('defend_the_center.wad')

    game.init()
    possible_actions = np.identity(3, dtype=int).tolist()
    return game, possible_actions

def test_environment():
    game, possible_actions = create_environment()
    episodes = 1

    for _ in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            img = state.screen_buffer # 当前游戏画面, 2D array
            misc = state.game_variables # [50.]
            action = random.choice(possible_actions)
            reward = game.make_action(action)
            print(action, 'reward:', reward)
            time.sleep(0.02)

        print('[*] Result:', game.get_total_reward())
        time.sleep(2)

    game.close()

def preprocess_frame(frame):
    cropped_frame = frame[40:, :]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, frame_size)
    return preprocessed_frame

def stack_frames(state, is_new_episode=False):
    global stacked_frames
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros(frame_size) for _ in range(stack_size)], maxlen=stack_size)

        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    return np.stack(stacked_frames, axis=2)

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


class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.0001, name='PGNetwork'):
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions = tf.placeholder(tf.float32, [None, action_size], name='actions')
            self.discounted_episode_rewards = tf.placeholder(tf.float32, [None, ], name='discounted_episode_rewards')
            self.mean_reward = tf.placeholder(tf.float32, name='mean_reward')

            conv1 = tf.layers.conv2d(
                inputs = self.inputs,
                filters = 32,
                kernel_size = [8, 8],
                strides = [4, 4],
                padding = 'VALID',
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = 'conv1'
            )
            conv1_batchnorm = tf.layers.batch_normalization(
                conv1,
                training = True,
                epsilon = 1e-5,
                name = 'conv1_batchnorm'
            )
            conv1_out = tf.nn.elu(conv1_batchnorm, name='conv1_out')

            conv2 = tf.layers.conv2d(
                inputs = conv1_out,
                filters = 64,
                kernel_size = [4, 4],
                strides = [2, 2],
                padding = 'VALID',
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = 'conv2'
            )
            conv2_batchnorm = tf.layers.batch_normalization(
                conv2,
                training = True,
                epsilon = 1e-5,
                name = 'conv2_batchnorm'
            )
            conv2_out = tf.nn.elu(conv2_batchnorm, name='conv2_out')

            conv3 = tf.layers.conv2d(
                inputs = conv2_out,
                filters = 128,
                kernel_size = [4, 4],
                strides = [2, 2],
                padding = 'VALID',
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = 'conv3'
            )
            conv3_batchnorm = tf.layers.batch_normalization(
                conv3,
                training = True,
                epsilon = 1e-5,
                name = 'conv3_batchnorm'
            )
            conv3_out = tf.nn.elu(conv3_batchnorm, name='conv3_out')

            flatten = tf.layers.flatten(conv3_out)
            fc1 = tf.layers.dense(
                inputs = flatten,
                units = 512,
                activation = tf.nn.elu,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = 'fc1'
            )
            fc2 = tf.layers.dense(
                inputs = fc1,
                units = action_size,
                activation = None,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = 'fc2'
            )
            self.output = tf.nn.softmax(fc2)

            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=self.actions)
            self.loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards)
            self.train = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)


def train():
    game, possible_actions = create_environment()

    # set hyperparameters
    ###########################################
    state_size = [*frame_size, stack_size]
    action_size = game.get_available_buttons_size()
    learning_rate = 0.0001
    total_episodes = 5000
    batch_size = 1000
    gamma = 0.99
    check_step = 5
    ###########################################

    # train PG
    ###########################################
    tf.reset_default_graph()
    PG = PGNetwork(state_size, action_size, learning_rate)

    writer = tf.summary.FileWriter('train_log')
    tf.summary.scalar('Loss', PG.loss)
    tf.summary.scalar('Reward mean', PG.mean_reward)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    all_rewards = []
    total_rewards = 0
    maximum_reward_recorded = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(1, total_episodes+1):
            episode_states, episode_actions, episode_rewards = [], [], []

            game.new_episode()
            state = game.get_state().screen_buffer
            state = stack_frames(state, True)

            while not game.is_episode_finished():
                state = game.get_state().screen_buffer
                state = stack_frames(state)

                action_prob = sess.run(PG.output, feed_dict={
                    PG.inputs: state.reshape((1, *state_size))
                })
                action = np.random.choice(range(action_size), p=action_prob.ravel())
                action = possible_actions[action]
                reward = game.make_action(action)

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)                

            episode_rewards_sum = np.sum(episode_rewards)
            all_rewards.append(episode_rewards_sum)
            total_rewards = np.sum(all_rewards)
            mean_reward = np.divide(total_rewards, episode + 1)
            maximum_reward_recorded = np.amax(all_rewards)


            episode_rewards = discount_and_normalize_rewards(episode_rewards, gamma)
            loss, _ = sess.run([PG.loss, PG.train], feed_dict={
                    PG.inputs: np.array(episode_states),
                    PG.actions: np.array(episode_actions),
                    PG.discounted_episode_rewards: episode_rewards
                })

            summary = sess.run(write_op, feed_dict={
                    PG.inputs: np.array(episode_states),
                    PG.actions: np.array(episode_actions),
                    PG.discounted_episode_rewards: episode_rewards,
                    PG.mean_reward: mean_reward
                })

            writer.add_summary(summary, episode)
            writer.flush()

            print('='*30)
            print('[*] Episode:', episode)
            print('[*] Reward:', episode_rewards_sum)
            print('[*] Mean Reward:', mean_reward)
            print('[*] Max reward so far:', maximum_reward_recorded)
            print('[*] Loss:', loss)

            if episode % check_step == 0:
                save_path = saver.save(sess, './model/model.ckpt')
                print('[*] Model Saved:', save_path)

    print('[*] Train done')
    game.close()
    ###########################################

def play():
    game, possible_actions = create_environment()

    state_size = [*frame_size, stack_size]
    action_size = game.get_available_buttons_size()
    PG = PGNetwork(state_size, action_size)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")

        game.new_episode()
        frame = game.get_state().screen_buffer
        state = stack_frames(frame, True)

        while not game.is_episode_finished():
            frame = game.get_state().screen_buffer
            state = stack_frames(frame)

            action_prob = sess.run(PG.output, feed_dict={
                PG.inputs: state.reshape((1, *state_size))
            })
            action = np.random.choice(range(action_size), p=action_prob.ravel())
            action = possible_actions[action]
            game.make_action(action)
        
        score = game.get_total_reward()
        print("[*] Score: ", score)

    game.close()

if __name__ == '__main__':
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play()
    elif sys.argv[1] == '--test':
        test_environment()
