import numpy as np
import random

def softmax(x):
    p = np.exp(x - np.max(x))
    p /= np.sum(p)
    return p

class Node:
    def __init__(self, action=None, parent=None, player=None, prior_p=1.0):
        self.action = action
        self.parent = parent
        self.childs = {}

        self.W = 0 # total action value
        self.N = 0 # visit count
        self.Q = 0 # mean action value
        self.P = prior_p # prior probability of selecting that edge

        self.current_player = player
        self.next_player = 3 - player

    def select(self):
        # You should think carefully why use -Q here
        return max(self.childs.values(), key=lambda c: -c.Q + c.U())

    def expand(self, actions, probs):
        for i in range(len(actions)):
            a, p = actions[i], probs[i]
            n = Node(a, self, self.next_player, p)
            self.childs[a] = n
    
    def update(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N
        # self.Q += (v - self.Q) / self.N

    def back_propagate(self, v):
        self.update(v)
        if self.parent:
            self.parent.back_propagate(-v)

    def U(self, c_puct=5.0):
        # c_puct -- a number in (0, inf) controlling the relative impact of
        #       values (Q) and prior probability (P) on this node's score
        #       it is a constant determining the level of exploration
        if self.parent:
            return c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return 0

    def __repr__(self):
        return "[A: %s, P: %.2f, Q+U: %.2f, W/N: %.1f/%d]" \
        % (self.action, self.P, self.Q + self.U(), self.W, self.N)

    def show_node_tree(self, indent=0):
        print("|  " * indent + str(self))

        for c in self.childs.values():
            c.show_node_tree(indent+1)

    def show_children_nodes(self):
        print('\n[*] Child Nodes')
        for c in self.childs.values(): print(c)


class MCTS:
    def __init__(self, neural_network_fn, playout_itermax, playout_depth=4):
        self.f = neural_network_fn # (p, v) = f(s)
        self.playout_itermax = playout_itermax
        self.playout_depth = playout_depth

    def set_rootnode(self, starting_player):
        self.rootnode = Node(player=starting_player)

    def get_move(self, state, stochastically=False, show_node=False, verbose=False):
        for i in range(self.playout_itermax):
            self.playout(self.rootnode, state.clone())

        if show_node:
            if verbose: self.rootnode.show_node_tree()
            else: self.rootnode.show_children_nodes()

        action_probs = np.zeros((state.w, state.w))
        acts, probs = [], []
        for c in self.rootnode.childs.values():
            acts.append(c.action)
            probs.append(c.N)
        probs = softmax(probs)
        for a, p in zip(acts, probs):
            action_probs[a] = p
        action_probs = action_probs.flatten()

        if stochastically:
            # add Dirichlet Noise for exploration (for self-play training)
            epsilon = 0.25
            eta = 0.3 # Dirichlet noise
            i = np.random.choice(
                len(acts),
                p = (1 - epsilon) * probs + epsilon * np.random.dirichlet(eta * np.ones(len(probs)))
                )
            action = acts[i]
        else: # deterministically, for competitive play
            action = max(self.rootnode.childs.values(), key=lambda c: c.N).action

        return action, action_probs

    def update_with_move(self, action, state):
        if action in self.rootnode.childs:
            self.rootnode = self.rootnode.childs[action]
            self.rootnode.parent = None
        else:
            self.rootnode = Node(player=state.player_just_moved)

    def playout(self, node, state):
        #=======================================
        # MCTS without neural network
        #=======================================
        # # Select & Expand
        # for i in range(self.playout_depth):
        #     if node.childs == {}:
        #         node.expand(state.actions())
        #
        #     node = node.select()
        #     state.move(node.action)
        #
        # # Rollout
        # self.rollout(state)
        #
        # # Backpropagate
        # while node != None:
        #     node.update(state.reward(node.player_just_moved))
        #     node = node.parent
        #=======================================


        # Select
        while 1:
            if node.childs == {}:
                break

            node = node.select()
            state.move(node.action)

        # Rollout
        v, a, p = self.evaluate_state(state)

        if state.is_end:
            v = -1
        else:
            # Expand
            node.expand(a, p)

        # Backpropagate
        node.back_propagate(v)


    def evaluate_state(self, state):
        x = state.nn_input.reshape((1, *state.nn_input.shape))
        value, probs = self.f.pred(x)
        v = value[0, 0]
        a = state.actions()
        p = []
        probs = probs.reshape((state.w, state.w))
        for i, j in a:
            p.append(probs[i, j])
        p = np.array(p)
        if p.sum() > 0: p /= p.sum()
        return v, a, p

    def rollout(self, state):
        while not state.is_end:
            state.move(random.choice(state.actions()))

