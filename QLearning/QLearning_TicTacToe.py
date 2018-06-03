import game
import numpy as np
import random

class RandomPlayer():
    def __init__(self):
        self.name = 'Random'
        self.win_n = 0

    def action(self, state, actions):
        return random.choice(actions)

    def reward(self, reward, state):
        if reward == 1:
            self.win_n += 1

    def episode_end(self, episode):
        pass

class QLearningPlayer():
    def __init__(self):
        self.name = 'Q-Learning'
        self.q = {}
        self.init_q = 1 # "optimistic" 1.0 initial values
        self.lr = 0.3
        self.gamma = 0.9
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.01
        self.action_n = 9
        self.win_n = 0

        self.last_state = (' ',) * 9
        self.last_action = -1

    def action(self, state, actions):
        state = tuple(state)
        self.last_state = state

        r = random.uniform(0, 1)
        if r > self.epsilon:
            if self.q.get(state):
                i = np.argmax([self.q[state][a] for a in actions])
                action = actions[i]
            else:
                self.q[state] = [self.init_q] * self.action_n
                action = random.choice(actions)
        else:
            action = random.choice(actions)

        self.last_action = action
        return action

    def reward(self, reward, state):
        if self.last_action >= 0:
            if reward == 1:
                self.win_n += 1

            state = tuple(state)
            if self.q.get(self.last_state):
                q = self.q[self.last_state][self.last_action]
            else:
                self.q[self.last_state] = [self.init_q] * self.action_n
                q = self.init_q

            self.q[self.last_state][self.last_action] = q + self.lr * (reward + self.gamma * np.max(self.q.get(state, [self.init_q]*self.action_n)) - q)

    def episode_end(self, episode):
        # epsilon decay
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*(episode+1))

    def print_q(self):
        for k,v in self.q.items():
            print(k,v)

class HumanPlayer():
    def __init__(self):
        self.name = 'Human'

    def action(self, state, actions):
        a = int(input('your move:')) - 1
        return a


def train(trails_num, p1, p2, env):
    for episode in range(trails_num):
        
        state, win, done, info = env.reset(X=p1, O=p2)

        for (cur_player, oth_player) in env.player_turn():
            #env.render()
            action = cur_player.action(state, env.action_space)
            state, win, done, info = env.step(action)

            if done:
                if win:
                    cur_player.reward(1, state)
                    oth_player.reward(-1, state)
                else:
                    cur_player.reward(0.5, state)
                    oth_player.reward(0.5, state)
                #env.render()
                break
            else:
                oth_player.reward(0, state)
        
        env.playerX.episode_end(episode)
        env.playerO.episode_end(episode)
    
    print('='*20)
    print('Train result - %d episodes' % trails_num)
    print('{} win rate: {}'.format(p1.name, p1.win_n / trails_num))
    print('{} win rate: {}'.format(p2.name, p2.win_n / trails_num))
    print('players draw rate: {}'.format((trails_num - p1.win_n - p2.win_n) / trails_num))
    print('='*20)


def play(p1, p2, env):
    while 1:
        state, win, done, info = env.reset(X=p1, O=p2)
        for (cp, op) in env.player_turn():
            print()
            env.render()
            action = cp.action(state, env.action_space)
            state, win, done, info = env.step(action)
            if done:
                env.render()
                break

if __name__ == '__main__':
    env = game.make('TicTacToe')
    p1 = QLearningPlayer()
    p2 = QLearningPlayer()
    p3 = HumanPlayer()
    p4 = RandomPlayer()

    train(100000, p1, p4, env)
    print()
    print('Human play')
    print()

    play(p1, p3, env)
