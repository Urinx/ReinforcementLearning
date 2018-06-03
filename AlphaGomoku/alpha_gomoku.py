import sys
import time
import random
import numpy as np
from collections import deque
from gomoku import Gomoku
from mcts import MCTS
from neural_network import Residual_CNN, Simple_CNN

#======================
# Configuration
#======================
# 8x8
game_board_width = 8
mcts_playout_itermax_train = 400
mcts_playout_itermax_play = 1000
model_file = 'Simple_CNN_8x8_3000'
policy_network = Simple_CNN # or Residual_CNN
#======================
# 19x19
# game_board_width = 19
# mcts_playout_itermax_train = 1000
# mcts_playout_itermax_play = 2000
# model_file = 'Simple_CNN_19x19_3000'
# policy_network = Simple_CNN
#======================

def random_play(game):
    return random.choice(game.actions())

def human_play():
    t = input('[*] Your turn (i j): ')
    a, b = t.split(' ')
    i, j = int(a), int(b)
    return (i, j)

def play_game():
    game = Gomoku(game_board_width)
    policy = policy_network(input_dim=game.nn_input.shape, output_dim=game.w**2)
    policy.load(model_file)
    mcts_player = MCTS(policy, mcts_playout_itermax_play)

    starting_player = random.choice([1,2])
    game.reset(starting_player)
    mcts_player.set_rootnode(starting_player)
    while not game.is_end:
        print(game)
        # print(game.nn_input)

        if game.current_player == 1: # Player X
            action, _ = mcts_player.get_move(game)
        else: # Player O
            action = human_play()
        
        game.move(action)
        mcts_player.update_with_move(action, game)

        print("[*] Player %s move: %s\n" % (['X', 'O'][game.player_just_moved-1], action))

    print(game)
    if game.winner > 0:
        print("[*] Player %s win" % ['X', 'O'][game.winner-1])
    else:
        print("[*] Player draw")

def self_play(game, player, render=False):
    starting_player = random.choice([1,2])
    game.reset(starting_player)
    player.set_rootnode(starting_player)
    board_states, mcts_probs, cur_players = [], [], []

    while not game.is_end:
        if render: print(game)

        action, action_probs = player.get_move(game, stochastically=True, show_node=render)

        board_states.append(game.nn_input)
        mcts_probs.append(action_probs)
        cur_players.append(game.current_player)

        game.move(action)
        player.update_with_move(action, game)

        if render: print("[*] Player %s move: %s\n" % (['X', 'O'][game.player_just_moved-1], action))

    rewards = list(map(game.reward, cur_players))

    if render:
        print(game)
        if game.winner > 0:
            print("[*] Player %s win" % ['X', 'O'][game.winner-1])
        else:
            print("[*] Player draw")

    return list(zip(board_states, mcts_probs, rewards)), game.winner, starting_player

def augment_data(play_data):
    # augment the data set by rotation and flipping
    extend_data = []
    for state, pi, z in play_data:
        w = state.shape[-1]

        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_pi = np.rot90(pi.reshape((w, w)), i)
            extend_data.append((equi_state, equi_pi.flatten(), z))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_pi =np.fliplr(equi_pi)
            extend_data.append((equi_state, equi_pi.flatten(), z))

    return extend_data
    

def train():
    game_episode_num = 3000
    selfplay_batch_size = 1
    data_buffer_size = 10000
    check_step = 10
    train_batch_size = 512

    data_buffer = deque(maxlen=data_buffer_size)

    game = Gomoku(game_board_width)
    policy = policy_network(input_dim=game.nn_input.shape, output_dim=game.w**2)
    mcts_player = MCTS(policy, mcts_playout_itermax_train)
    winner_num = [0] * 3

    print('[*] Start self play')
    # game episode
    for i in range(game_episode_num):

        # get train data
        start_time = time.time()
        for _ in range(selfplay_batch_size):
            play_data, winner, starting_player = self_play(game, mcts_player)
            episode_len = len(play_data)
            extend_data = augment_data(play_data)
            data_num = len(extend_data)
            data_buffer.extend(extend_data)
            winner_num[winner] += 1
        end_time = time.time()

        print('[*] Episode: {}, length: {}, start: {}, winner: {}, data: {}, time: {}s, win ratio: X {:.1f}%, O {:.1f}%, - {:.1f}%'.format(
            i+1, episode_len, ['-', 'X', 'O'][starting_player], ['-', 'X', 'O'][winner], data_num, int(end_time - start_time),
            winner_num[1] / (i+1) * selfplay_batch_size * 100,
            winner_num[2] / (i+1) * selfplay_batch_size * 100,
            winner_num[0] / (i+1) * selfplay_batch_size * 100,
        ))

        # train
        if len(data_buffer) > train_batch_size:
            mini_batch = random.sample(data_buffer, train_batch_size)
            state_batch = np.array([d[0] for d in mini_batch])
            pi_batch = np.array([d[1] for d in mini_batch])
            z_batch = np.array([d[2] for d in mini_batch])

            policy.train(state_batch, [z_batch, pi_batch])

        # check current policy model and save the params
        if (i + 1) % check_step == 0:
            policy.loss_history.plot_loss('loss.png')
            print('[*] Save current policy model')
            policy.save(model_file)
            print('[*] done')



if __name__ == "__main__":
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play_game()
