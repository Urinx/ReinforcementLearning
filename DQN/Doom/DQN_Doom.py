import tensorflow as tf
import numpy as np
from vizdoom import DoomGame
import random
import time
from skimage import transform
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def create_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions

def test_environment():
    game, actions = create_environment()
    episodes = 1

    for _ in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            img = state.screen_buffer # 当前游戏画面, 2D array
            misc = state.game_variables # [50.]
            action = random.choice(actions)
            reward = game.make_action(action)
            print(action, 'reward:', reward)
            time.sleep(0.02)

        print('[*] Result:', game.get_total_reward())
        time.sleep(2)

    game.close()


def preprocess_frame(state):
    cropped_frame = state[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


def stack_states(stacked_frames, state):
    frame = preprocess_frame(state)
    stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state


class build_DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # 84x84x4
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions = tf.placeholder(tf.float32, [None, action_size], name='actions')
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            # 20x20x32
            self.conv1 = tf.layers.conv2d(inputs = self.inputs,
                                        filters = 32,
                                        kernel_size = [8, 8],
                                        strides = [4, 4],
                                        padding = 'VALID',
                                        kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                        name = 'conv1')
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                        training = True,
                                        epsilon = 1e-5,
                                        name = 'batch_norm1')
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name='conv1_out')

            # 9x9x64
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                        filters = 64,
                                        kernel_size = [4, 4],
                                        strides = [2, 2],
                                        padding = 'VALID',
                                        kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                        name = 'conv2')
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                        training = True,
                                        epsilon = 1e-5,
                                        name = 'batch_norm2')
            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name='conv2_out')

            # 3x3x128
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                        filters = 128,
                                        kernel_size = [4, 4],
                                        strides = [2, 2],
                                        padding = 'VALID',
                                        kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                        name = 'conv3')
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                        training = True,
                                        epsilon = 1e-5,
                                        name = 'batch_norm3')
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name='conv3_out')

            # 1152
            self.flatten = tf.layers.flatten(self.conv3_out)
            # 512
            self.fc = tf.layers.dense(inputs = self.flatten,
                                    units = 512,
                                    activation = tf.nn.elu,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                    name = 'fc1')
            # 3
            self.output = tf.layers.dense(inputs = self.fc,
                                    units = 3,
                                    activation = None,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                    name = 'output')

            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
            # # The loss is the difference between our predicted Q and the Q_target
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


def train():
    game, possible_actions = create_environment()

    # Set Hyperparameters
    #####################
    state_size = [84, 84, 4]
    action_size = game.get_available_buttons_size()
    learning_rate = 0.0002

    total_episodes = 5000
    max_steps = 100
    batch_size = 64

    explore_max = 1.0
    explore_min = 0.01
    decay_rate = 0.0001
    gamma = 0.99

    pretrain_length = batch_size
    memory_size = 50000
    stack_size = 4

    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)],
                            maxlen=stack_size)
    memory = Memory(max_size=memory_size)
    #####################


    # make pretrain samples
    ###########################################
    game.new_episode()

    for i in range(pretrain_length):
        if i == 0:
            state = game.get_state().screen_buffer
            state = stack_states(stacked_frames, state)

        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            game.new_episode()
        else:
            next_state = game.get_state().screen_buffer
            next_state = stack_states(stacked_frames, next_state)
            memory.add((state, action, reward, next_state, done))

            state = next_state
    ###########################################


    # train deep Q neural network
    ###########################################
    tf.reset_default_graph()
    DQNetwork = build_DQNetwork(state_size, action_size, learning_rate)

    writer = tf.summary.FileWriter('train_log')
    tf.summary.scalar('loss', DQNetwork.loss)
    saver = tf.train.Saver()

    rewards_list = []
    decay_step = 0
    game.init()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for episode in range(total_episodes):
            game.new_episode()

            step = 0
            frame = game.get_state().screen_buffer
            state = stack_states(stacked_frames, frame)

            while step < max_steps:
                step += 1
                decay_step += 1

                exp_exp_tradeoff = np.random.rand()
                explore_probability = explore_min + (explore_max - explore_min) * np.exp(-decay_rate * decay_step)

                if explore_probability > exp_exp_tradeoff:
                    action = random.choice(possible_actions)
                else:
                    Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: state.reshape(1, *state.shape)})
                    action = possible_actions[int(np.argmax(Qs))]

                reward = game.make_action(action)
                done = game.is_episode_finished()

                if done:
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state = stack_states(stacked_frames, next_state)
                    total_reward = game.get_total_reward()
                    formated_str = 'Episode: {}, Total reward: {}, Training loss: {:.4f}, Explore P: {:.4f}'
                    print(formated_str.format(episode, total_reward, loss, explore_probability))

                    rewards_list.append((episode, total_reward))
                    memory.add((state, action, reward, next_state, done))
                    step = max_steps
                else:
                    next_state = game.get_state().screen_buffer
                    next_state = stack_states(stacked_frames, next_state)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state

                # train DQNetwork == update Qtable
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch], ndmin=3)
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                dones = np.array([each[4] for each in batch])

                target_Qs_batch = []
                target_Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs: next_states})

                for i in range(batch_size):
                    terminal = dones[i]

                    if terminal:
                        target_Qs_batch.append(rewards[i])
                    else:
                        target = rewards[i] + gamma * np.max(target_Qs[i])
                        target_Qs_batch.append(target)

                targets = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.inputs: states,
                                               DQNetwork.target_Q: targets,
                                               DQNetwork.actions: actions})

                # Write TF Summaries
                summary = sess.run(tf.summary.merge_all(),
                                feed_dict={DQNetwork.inputs: states,
                                            DQNetwork.target_Q: targets,
                                            DQNetwork.actions: actions})
                writer.add_summary(summary, episode)
                writer.flush()

            if episode % 5 == 0:
                save_path = saver.save(sess, './model/model.ckpt')
                print('[*] Model Saved:', save_path)
    print('Train done')
###########################################


def play():
    with tf.Session() as sess:
        state_size = [84, 84, 4]
        action_size = 3
        learning_rate = 0.0002
        DQNetwork = build_DQNetwork(state_size, action_size, learning_rate)

        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")

        game, possible_actions = create_environment()
        totalScore = 0
        episodes = 10
        stack_size = 4
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)],
                            maxlen=stack_size)

        for i in range(episodes):
            game.new_episode()

            while not game.is_episode_finished():
                frame = game.get_state().screen_buffer
                state = stack_states(stacked_frames, frame)

                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))})
                action = possible_actions[int(np.argmax(Qs))]
                game.make_action(action)
            
            score = game.get_total_reward()
            print("Episode {} Score: {}".format(i, score))
            totalScore += score

        print("[*] Average Score: ", totalScore / episodes)
        game.close()


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play()

