import sys
import time
import random
import numpy as np
from collections import deque

from rna import RNA
from mcts import MCTS
from neural_network import Simple_CNN

#======================
# Configuration
#======================
rna_data = {
    'id': 'None',
    'len': 30,
    'seq': 'AAGCGGAACGAAACGUUGCUUUUGCGCCCU',
    'pairs': [
        (1, 29), (2, 28), (4, 27), (5, 26), (7, 15),
        (8, 14), (9, 13), (16, 25), (17, 24), (18, 23)
    ],
    'sec': '.((.((.(((...)))(((....)))))))'
}
train_episode_num = 1000
train_batch_size = 512
train_epochs = 1
train_check_step = 10
train_data_buffer_size = 10000
mcts_playout_itermax_train = 2000
model_file = 'Simple_CNN_720'
policy_network = Simple_CNN
#======================

def rna_folding(rna, mcts, stochastically=True, render=False):
    rna.reset()
    mcts.reset()

    rna_states, mcts_probs = [], []
    min_energy = rna.energy()
    if render: print(rna)

    while rna.action_space:
        action, action_probs = mcts.get_action(rna, stochastically=True, show_node=render)

        rna_states.append(rna.state)
        mcts_probs.append(action_probs)

        rna.move(action)
        mcts.update_with_action(action)

        energy = rna.energy()
        if energy < min_energy:
            min_energy = energy

        if render:
            print("[*] RNA pair position: %s" % (action,))
            print("[*] RNA secondary structure: %s" % ''.join(rna.sec))
            print("[*] Current energy: %.2f" % energy)
            print("[*] Min energy: %.2f\n" % min_energy)
            print(rna)

    final_energy = rna.energy()

    return list(zip(rna_states, mcts_probs, [final_energy] * len(rna_states))), final_energy

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
    data_buffer = deque(maxlen=train_data_buffer_size)
    rna = RNA(rna_data['seq'], rna_data['pairs'])
    policy = policy_network(input_dim=rna.state.shape, output_dim=rna.len**2)
    policy.load(model_file)
    mcts = MCTS(policy, mcts_playout_itermax_train)
    render = False

    print('[*] RNA seq:', rna.seq)
    print('[*] RNA native sec:', rna_data['sec'])
    print('[*] Start RNA folding train')

    for i in range(train_episode_num):
        # get train data
        start_time = time.time()
        folding_data, final_energy = rna_folding(rna, mcts, render=render)

        episode_len = len(folding_data)
        extend_data = augment_data(folding_data)
        data_num = len(extend_data)
        data_buffer.extend(extend_data)
        end_time = time.time()

        print('[*] Episode: {}, length: {}, final energy: {}, data: {}, time: {}s\n    find pairs: {}'.format(
            i+1, episode_len, final_energy, data_num, int(end_time - start_time), rna.find_pairs
        ))

        # train
        if len(data_buffer) > train_batch_size:
            mini_batch = random.sample(data_buffer, train_batch_size)
            state_batch = np.array([d[0] for d in mini_batch])
            pi_batch = np.array([d[1] for d in mini_batch])
            z_batch = np.array([d[2] for d in mini_batch])

            policy.train(state_batch, [z_batch, pi_batch], train_epochs)

        # check current policy model and save the params
        if (i + 1) % train_check_step == 0:
            policy.loss_history.plot_loss('loss.png')
            print('[*] Save current policy model')
            policy.save(model_file)
            print('[*] done')


def play():
    rna = RNA(rna_data['seq'], rna_data['pairs'])
    policy = policy_network(input_dim=rna.state.shape, output_dim=rna.len**2)
    policy.load(model_file)
    mcts = MCTS(policy, mcts_playout_itermax_train)
    render = True

    print('[*] RNA seq:', rna.seq)
    print('[*] RNA native sec:', rna_data['sec'])
    print('[*] Start RNA folding train')

    rna_folding(rna, mcts, stochastically=False, render=render)


if __name__ == "__main__":
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play()
