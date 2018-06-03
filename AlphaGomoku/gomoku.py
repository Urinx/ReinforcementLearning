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
        self.w = w
        self.reset()
    
    def reset(self, current_player=1):
        w = self.w
        self.current_player = current_player
        self.first_player = current_player
        self.player_just_moved = 3 - current_player
        self.board = [] # 0 = empty, 1 = player 1 (X), 2 = player 2 (O)

        for y in range(w):
            self.board.append([0] * w)
        self.is_end = False
        self.winner = -1 # 0 = draw, 1 = player 1 (X), 2 = player 2 (O)

        # 1 if stone here and 0 if stone not here
        # fisrt 1 stack - position of current player's stones
        # next 1 stack - position of last player's stones
        # next 1 stack - position of last player's move
        # last 1 stack - All 1 if it's first play, all 0 if it's second play
        self.nn_input = np.zeros((4, w, w))
        self.nn_input[-1] = 1

    def clone(self):
        st = Gomoku()
        st.w = self.w
        st.current_player = self.current_player
        st.first_player = self.first_player
        st.player_just_moved = self.player_just_moved
        st.board = [self.board[i][:] for i in range(self.w)]
        st.nn_input = np.copy(self.nn_input)
        return st

    def move(self, action):
        a, b = action
        assert 0 <= a <= self.w and 0 <= b <= self.w and self.board[a][b] == 0
        self.board[a][b] = self.current_player
        self.player_just_moved = self.current_player
        self.current_player = 3 - self.current_player
        self.check_end(action)

        # update nn_input
        self.nn_input = np.zeros((4, self.w, self.w))
        for i in range(self.w):
            for j in range(self.w):
                s = self.board[i][j]
                if s == self.current_player: self.nn_input[0, i, j] = 1
                elif s == self.player_just_moved: self.nn_input[1, i, j] = 1
        self.nn_input[2, a, b] = 1
        self.nn_input[3] = 1 if self.current_player == self.first_player else 0
        
    def actions(self):
        return [(i, j) for i in range(self.w) for j in range(self.w) if self.board[i][j] == 0]
    
    def check_end(self, action):
        a, b = action
        for i in range(5):
            if i <= a <= self.w - (5 - i) and i <= b <= self.w - (5 - i):
                if self.board[a-i][b-i] == self.board[a-i+1][b-i+1] == self.board[a-i+2][b-i+2] == self.board[a-i+3][b-i+3] == self.board[a-i+4][b-i+4]:
                    self.is_end = True
                    self.winner = self.player_just_moved
                    return
            if i <= a <= self.w - (5 - i):
                if self.board[a-i][b] == self.board[a-i+1][b] == self.board[a-i+2][b] == self.board[a-i+3][b] == self.board[a-i+4][b]:
                    self.is_end = True
                    self.winner = self.player_just_moved
                    return
            if i <= a <= self.w - (5 - i) and (4 - i) <= b <= (self.w - i - 1):
                if self.board[a-i][b+i] == self.board[a-i+1][b+i-1] == self.board[a-i+2][b+i-2] == self.board[a-i+3][b+i-3] == self.board[a-i+4][b+i-4]:
                    self.is_end = True
                    self.winner = self.player_just_moved
                    return
            if i <= b <= self.w - (5 - i):
                if self.board[a][b-i] == self.board[a][b-i+1] == self.board[a][b-i+2] == self.board[a][b-i+3] == self.board[a][b-i+4]:
                    self.is_end = True
                    self.winner = self.player_just_moved
                    return

        if self.actions() == []:
            self.is_end = True
            self.winner = 0

    def reward(self, player):
        if self.winner == 0: # tie
            return 0
        if self.winner == player:
            return 1
        elif self.winner == 3 - player:
            return -1
        if self.winner == -1:
            return 0

    def __repr__(self):
        row = '{:>2}  ' + ' | '.join(['{}'] * self.w) + ' '
        line = '\n   ' + ('----' * self.w)[:-1] + '\n'
        s = '   ' + '%2d  ' * self.w % tuple(range(self.w)) + '\n'
        s += line.join([row.format(i, *map(lambda j: [' ', 'X', 'O'][j], self.board[i])) for i in range(self.w)])
        return s
