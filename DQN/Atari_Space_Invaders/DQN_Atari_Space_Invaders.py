import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import random
import sys
import time

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


###########################################
# Constant
stack_size = 4
frame_size = (110, 84)
# Global variables
stacked_frames = deque([np.zeros(frame_size) for _ in range(stack_size)], maxlen=stack_size)
###########################################

def create_environment():
    env = retro.make(game='SpaceInvaders-Atari2600')
    possible_actions = np.array(np.identity(env.action_space.n, dtype=np.int).tolist())
    return env, possible_actions

def test_environment():
    env, possible_actions = create_environment()
    episodes = 1

    for _ in range(episodes):
        env.reset()
        done = False

        while not done:
            env.render()
            choice = random.randint(0, action_size - 1)
            action = possible_actions[choice]
            state, reward, done, info = env.step(action)

    env.close()

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(cropped_frame, frame_size)
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


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions = tf.placeholder(tf.float32, [None, action_size], name='actions')
            self.target_q = tf.placeholder(tf.float32, [None], name='target_q')

            conv1 = tf.layers.conv2d(
                inputs = self.inputs,
                filters = 32,
                kernel_size = [8, 8],
                strides = [4, 4],
                padding = 'VALID',
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = 'conv1'
            )
            conv1_out = tf.nn.elu(conv1, name='conv1_out')

            conv2 = tf.layers.conv2d(
                inputs = conv1_out,
                filters = 64,
                kernel_size = [4, 4],
                strides = [2, 2],
                padding = 'VALID',
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = 'conv2'
            )
            conv2_out = tf.nn.elu(conv2, name='conv2_out')

            conv3 = tf.layers.conv2d(
                inputs = conv2_out,
                filters = 64,
                kernel_size = [3, 3],
                strides = [2, 2],
                padding = 'VALID',
                kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name = 'conv3'
            )
            conv3_out = tf.nn.elu(conv3, name='conv3_out')

            flatten = tf.contrib.layers.flatten(conv3_out)
            fc = tf.layers.dense(
                inputs = flatten,
                units = 512,
                activation = tf.nn.elu,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = 'fc'
            )
            self.output = tf.layers.dense(
                inputs = fc,
                units = action_size,
                activation = None,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = 'output'
            )

            self.q = tf.reduce_sum(tf.multiply(self.output, self.actions))
            self.loss = tf.reduce_mean(tf.square(self.target_q - self.q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size = batch_size,
            replace = False
        )
        return [self.buffer[i] for i in index]


def train():
    env, possible_actions = create_environment()

    # set hyperparameters
    ###########################################
    state_size = [*frame_size, stack_size]
    action_size = env.action_space.n
    learning_rate = 0.00025
    total_episodes = 100
    check_step = 5
    max_steps = 50000
    batch_size = 64
    explore_start = 1.0
    explore_stop = 0.01
    decay_rate = 0.00001
    gamma = 0.9
    pretrain_length = batch_size
    memory_size = 1000000
    ###########################################


    # pre-populate train samples
    ###########################################
    memory = Memory(max_size=memory_size)
    state = env.reset()
    state = stack_frames(state, True)

    for i in range(pretrain_length):
        choice = random.randint(0, action_size - 1)
        action = possible_actions[choice]
        new_state, reward, done, _ = env.step(action)
        new_state = stack_frames(new_state)

        if done:
            new_state = np.zeros(state.shape)
            memory.add((state, action, reward, new_state, done))
            state = env.reset()
            state = stack_frames(state, True)
        else:
            memory.add((state, action, reward, new_state, done))
            state = new_state
    ###########################################

    # train DQN
    ###########################################
    tf.reset_default_graph()
    DQN = DQNetwork(state_size, action_size, learning_rate)

    writer = tf.summary.FileWriter('train_log')
    tf.summary.scalar('Loss', DQN.loss)
    write_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        decay_step = 0
        loss = None

        for episode in range(1, total_episodes+1):
            step = 0
            episode_rewards = []

            state = env.reset()
            state = stack_frames(state, True)

            while step < max_steps:
                step += 1
                decay_step += 1

                exp_exp_tradeoff = np.random.rand()
                explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

                if explore_probability > exp_exp_tradeoff:
                    choice = random.randint(0, action_size - 1)
                else:
                    qs = sess.run(DQN.output, feed_dict={
                            DQN.inputs: state.reshape((1, *state.shape))
                        })
                    choice = np.argmax(qs)

                action = possible_actions[choice]
                new_state, reward, done, _ = env.step(action)

                env.render()
                episode_rewards.append(reward)

                if done:
                    total_reward = np.sum(episode_rewards)

                    new_state = np.zeros(frame_size)
                    new_state = stack_frames(new_state)
                    memory.add((state, action, reward, new_state, done))

                    print(
                        '[*] Episode: {}, total reward: {}, explore p: {:.4f}, train loss: {:.4f}'.format(
                            episode, total_reward, explore_probability, loss
                        )
                    )
                    break
                else:
                    new_state = stack_frames(new_state)
                    memory.add((state, action, reward, new_state, done))
                    state = new_state

                # learning part
                ################
                batch = memory.sample(batch_size)
                states_mb = np.array([b[0] for b in batch], ndmin=3)
                actions_mb = np.array([b[1] for b in batch])
                rewards_mb = np.array([b[2] for b in batch])
                new_states_mb = np.array([b[3] for b in batch], ndmin=3)
                dones_mb = np.array([b[4] for b in batch])

                target_q_mb = []
                new_state_q_mb = sess.run(DQN.output, feed_dict={
                        DQN.inputs: new_states_mb,
                    })

                for i in range(batch_size):
                    is_done = dones_mb[i]
                    if is_done:
                        target_q_mb.append(rewards_mb[i])
                    else:
                        t = rewards_mb[i] + gamma * np.max(new_state_q_mb)
                        target_q_mb.append(t)

                target_q_mb = np.array(target_q_mb)

                loss, _ = sess.run([DQN.loss, DQN.optimizer], feed_dict={
                        DQN.inputs: states_mb,
                        DQN.actions: actions_mb,
                        DQN.target_q: target_q_mb
                    })

                summary = sess.run(write_op, feed_dict={
                        DQN.inputs: states_mb,
                        DQN.actions: actions_mb,
                        DQN.target_q: target_q_mb
                    })
                writer.add_summary(summary, episode)
                writer.flush()
                ################

            if episode % check_step == 0:
                save_path = saver.save(sess, './model/model.ckpt')
                print('[*] Model Saved:', save_path)

    print('[*] Train done')
    env.close()
    ###########################################


def play():
    env, possible_actions = create_environment()

    with tf.Session() as sess:
        total_rewards = 0

        state_size = [*frame_size, stack_size]
        action_size = env.action_space.n
        learning_rate = 0.00025
        DQN = DQNetwork(state_size, action_size, learning_rate)

        saver = tf.train.Saver()
        saver.restore(sess, './model/model.ckpt')

        # start game
        state = env.reset()
        state = stack_frames(state, True)
        done = False

        while not done:
            state_q = sess.run(DQN.output, feed_dict={
                    DQN.inputs: state.reshape((1, *state.shape))
                })
            choice = np.argmax(state_q)
            action = possible_actions[choice]
            new_state, reward, done, _ = env.step(action)
            
            env.render()
            total_rewards += reward
            state = stack_frames(new_state)

        print('[*] total score:', total_rewards)

    env.close()


if __name__ == '__main__':
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play()
    elif sys.argv[1] == '--test':
        test_environment()















