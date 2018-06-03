from math import *
import random

class Game:
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

class TicTacToe(Game):
    def __init__(self):
        self.player_just_moved = 2
        self.board = [0] * 9 # 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
        
    def clone(self):
        st = TicTacToe()
        st.player_just_moved = self.player_just_moved
        st.board = self.board[:]
        return st

    def move(self, action):
        assert action >= 0 and action <= 8 and action == int(action) and self.board[action] == 0
        self.player_just_moved = 3 - self.player_just_moved
        self.board[action] = self.player_just_moved
        
    def actions(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def win(self, player):
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == player:
                    return 1
                else:
                    return 0
        if self.actions() == []: return 0.5 # draw

    def end(self):
        return self.actions() == [] or self.win(1) == 1 or self.win(2) == 1

    def __repr__(self):
        line = '\n-----------\n'
        row = " {} | {} | {}"
        s = (row + line + row + line + row).format(*map(lambda i: [' ', 'X', 'O'][i], self.board))
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
                
def play_game():
    game = TicTacToe()

    while not game.end():
        print(game)

        if game.player_just_moved == 1:
            action = UCT(game, 1000) # Player O
        else:
            action = UCT(game, 100) # Player X
        
        game.move(action)
        print("[*] Player %s move: %d\n" % (['X', 'O'][game.player_just_moved-1], action))

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
