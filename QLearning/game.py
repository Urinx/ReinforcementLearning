import random

def make(game_name):
    if game_name == 'TicTacToe':
        return TicTacToe()

class TicTacToe():

    def __init__(self):
        self.reset()

    def render(self):
        line = '\n-----------\n'
        row = " {} | {} | {}"
        print((row + line + row + line + row).format(*self.state))
        print(self.info)

    def step(self, action):
        #print(action)
        self.state[action] = self.cur_player
        self.action_space.remove(action)

        self.check_end()
        if self.is_end:
            if self.is_win:
                self.info = 'player{} win!'.format(self.cur_player)
            else:
                self.info = 'players draw'
        else:
            self.info = 'player{} turn'.format(self.cur_player)
        return (self.state, self.is_win, self.is_end, self.info)

    def reset(self, X=None, O=None):
        self.state = [' '] * 9
        self.action_space = list(range(9))
        self.is_end = False
        self.is_win = False
        self.info = 'new game'
        self.playerX = X
        self.playerO = O
        self.cur_player = random.choice(['O','X'])
        return (self.state, self.is_win, self.is_end, self.info)

    def player_turn(self):
        while 1:
            if self.cur_player == 'O':
                cur = self.playerO
                oth = self.playerX
            else:
                cur = self.playerX
                oth = self.playerO
            
            self.info = 'player{} turn'.format(self.cur_player) 
            yield (cur, oth)
            
            self.cur_player = 'OX'.replace(self.cur_player, '')

    def check_end(self):
        for a,b,c in [(0,1,2), (3,4,5), (6,7,8),
                      (0,3,6), (1,4,7), (2,5,8),
                      (0,4,8), (2,4,6)]:
            if self.cur_player == self.state[a] == self.state[b] == self.state[c]:
                self.is_win = True
                self.is_end = True
                return

        if not any([s == ' ' for s in self.state]):
            self.is_win = False
            self.is_end = True
            return

