from math import *
import random
import numpy as np

class GameState:
    def __init__(self):
        self.player_just_moved = 2
        
    def clone(self):
        st = GameState()
        st.player_just_moved = self.player_just_moved
        return st

    def move(self, action):
        self.player_just_moved = 3 - self.player_just_moved
        
    def actions(self):
        """ Get all possible moves from this state.
        """
    
    def win(self, player):
        """ Get the game result from the viewpoint of player. 
        """

    def end(self):
        """ Whether the game is end or not
        """

    def __repr__(self):
        pass

class Gomoku(GameState):
    def __init__(self, w=8): # 15x15
        self.player_just_moved = 2
        self.board = [] # 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
        self.w = w
        for y in range(w):
            self.board.append([0] * w)
        
    def clone(self):
        st = Gomoku()
        st.player_just_moved = self.player_just_moved
        st.board = [self.board[i][:] for i in range(self.w)]
        st.w = self.w
        return st

    def move(self, action):
        a, b = action
        assert 0 <= a <= self.w and 0 <= b <= self.w and self.board[a][b] == 0
        self.player_just_moved = 3 - self.player_just_moved
        self.board[a][b] = self.player_just_moved
        
    def actions(self):
        return [(i, j) for i in range(self.w) for j in range(self.w) if self.board[i][j] == 0]
    
    def check_five(self, i, j, player):
        if 2 <= i < self.w-2 and 2 <= j < self.w-2 and self.board[i-2][j-2] == self.board[i-1][j-1] == self.board[i][j] == self.board[i+1][j+1] == self.board[i+2][j+2] == player:
            return 1
        elif 2 <= j < self.w-2 and self.board[i][j-2] == self.board[i][j-1] == self.board[i][j] == self.board[i][j+1] == self.board[i][j+2] == player:
            return 1
        elif 2 <= i < self.w-2 and 2 <= j < self.w-2 and self.board[i+2][j-2] == self.board[i+1][j-1] == self.board[i][j] == self.board[i-1][j+1] == self.board[i-2][j+2] == player:
            return 1
        elif 2 <= i < self.w-2 and self.board[i-2][j] == self.board[i-1][j] == self.board[i][j] == self.board[i+1][j] == self.board[i+2][j] == player:
            return 1
        return 0

    def win(self, player):
        for i in range(self.w):
            for j in range(self.w):
                if self.check_five(i, j, player):
                    return 1
                elif self.check_five(i, j, 3-player):
                    return 0
        if self.actions() == []: return 0.5
        return -1

    def end(self):
        return self.win(1) >= 0

    def __repr__(self):
        row = '{:>2}  ' + ' | '.join(['{}'] * self.w) + ' '
        line = '\n   ' + ('----' * self.w)[:-1] + '\n'
        s = '   ' + '%2d  ' * self.w % tuple(range(self.w)) + '\n'
        s += line.join([row.format(i, *map(lambda j: [' ', 'X', 'O'][j], self.board[i])) for i in range(self.w)])
        return s

class Node:
    def __init__(self, action=None, parent=None, state=None):
        self.action = action
        self.parent = parent
        self.childs = []
        self.W = 0
        self.N = 0
        self.untried_actions = state.actions()
        self.player_just_moved = state.player_just_moved
        
    def select(self):
        s = sorted(self.childs, key = lambda c: c.U())[-1]
        return s
    
    def add_child(self, a, s):
        n = Node(a, self, s)
        self.untried_actions.remove(a)
        self.childs.append(n)
        return n
    
    def update(self, result):
        self.N += 1
        self.W += result

    def U(self):
        if self.parent:
            return self.W / self.N + sqrt(2 * log(self.parent.N) / self.N)
        return 0

    def __repr__(self):
        return "[A: %s, U: %.2f, W/N: %.1f/%d, Untried: %s]" \
        % (self.action, self.U(), self.W, self.N, self.untried_actions)

    def show_node_tree(self, indent=0):
        print("|  " * indent + str(self))

        for c in self.childs:
            c.show_node_tree(indent+1)

    def show_children_nodes(self):
        print('\n[*] Child Nodes')
        for c in self.childs: print(c)


def UCT(rootstate, itermax, verbose=False):
    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.clone()

        # Select
        while node.untried_actions == [] and node.childs != []:
            node = node.select()
            state.move(node.action)

        # Expand
        if node.untried_actions != []:
            action = random.choice(node.untried_actions)
            state.move(action)
            node = node.add_child(action, state)

        # Rollout
        while state.actions() != []:
            state.move(random.choice(state.actions()))

        # Backpropagate
        while node != None:
            node.update(state.win(node.player_just_moved))
            node = node.parent

    if verbose: rootnode.show_node_tree()
    else: rootnode.show_children_nodes()

    return sorted(rootnode.childs, key = lambda c: c.N)[-1].action

def random_play(game):
    return random.choice(game.actions())

def human_play():
    t = input('[*] Your turn (i j): ')
    a, b = t.split(' ')
    i, j = int(a), int(b)
    return (i, j)

def play_game():
    game = Gomoku()

    while not game.end():
        print(game)

        if game.player_just_moved == 1:
            # action = UCT(game, 1000) # Player O
            action = random_play(game)
        else:
            action = UCT(game, 10000) # Player X
            # action = human_play()
        
        game.move(action)
        print("[*] Player %s move: %s\n" % (['X', 'O'][game.player_just_moved-1], action))

    print(game)
    r = game.win(game.player_just_moved)
    if r == 1:
        print("[*] Player %s win" % ['X', 'O'][game.player_just_moved-1])
    elif r == 0:
        print("[*] Player %s win" % ['X', 'O'][2-game.player_just_moved])
    else:
        print("[*] Player draw")

if __name__ == "__main__":
    play_game()
