import numpy as np
import itertools

class RNA():
    def __init__(self, seq, pairs):
        self.seq = seq
        self.len = len(seq)
        self.native_pairs = pairs
        self.pair_rule = [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]

        self.nat_mat = np.zeros((self.len, self.len))
        for (i, j) in self.native_pairs:
            self.nat_mat[i, j] = 1
            self.nat_mat[j, i] = 1

        self.find_pairs = []
        self.sec = []
        self.action_space = []
        self.state = np.zeros((18, self.len, self.len))

        self.reset()
    
    def reset(self):
        L = self.len
        self.sec = ['.'] * L
        self.action_space = []
        self.find_pairs = []

        self.state = np.zeros((18, L, L))

        encode_list = list(itertools.product(('A','U','C','G'), repeat=2))

        for i in range(L):
            for j in range(i, L):
                a, b = self.seq[i], self.seq[j]
                h = encode_list.index((a, b))
                self.state[h, i, j] = 1
                self.state[h, j, i] = 1

                if j - i >= 4 and (a, b) in self.pair_rule:
                    self.action_space.append((i, j))
                    self.state[-2, i, j] = 1
                    self.state[-2, j, i] = 1

    def clone(self):
        rna = RNA(self.seq, self.native_pairs)
        rna.action_space = self.action_space[:]
        rna.find_pairs = self.find_pairs[:]
        rna.sec = self.sec[:]
        rna.state = np.copy(self.state)
        return rna

    def move(self, action):
        i, j = action

        # process the remain possible actions
        remain = []
        for a, b in self.action_space:
            if b < i:
                remain.append((a, b))
            elif a > j:
                remain.append((a, b))
            elif a < i and b > j:
                remain.append((a, b))
            elif a > i and b < j:
                remain.append((a, b))
        self.action_space = remain

        self.find_pairs.append(action)
        self.sec[i] = '('
        self.sec[j] = ')'

        # update state
        self.state[-1, i, j] = 1
        self.state[-1, j, i] = 1

        self.state[-2] = 0
        for a, b in remain:
            self.state[-2, a, b] = 1
            self.state[-2, b, a] = 1
    
    def energy(self):
        return ((self.nat_mat - self.state[-1]) ** 2).sum()


    def __repr__(self):
        L = self.len
        mat = np.zeros((L, L))
        for (i, j) in self.native_pairs:
            mat[j, i] = 1
        for i in range(L):
            for j in range(i+1, L):
                if self.state[-1, i, j]:
                    mat[i, j] = 2
                elif self.state[-2, i, j]:
                    mat[i, j] = 3

        row = ' {}  ' + ' | '.join(['{}'] * L) + ' '
        line = '\n   ' + ('----' * L)[:-1] + '\n'
        s = '   ' + ' %s  ' * L % tuple(self.seq) + '\n'
        s += line.join([row.format(self.seq[i], *map(lambda j: [' ', 'O', '\033[31mX\033[0m', 'X'][int(j)], mat[i])) for i in range(L)])
        return s
