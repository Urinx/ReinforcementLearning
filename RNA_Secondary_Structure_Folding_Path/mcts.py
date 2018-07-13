import numpy as np
import random

def softmax(x):
    p = np.exp(x - np.max(x))
    p /= np.sum(p)
    return p

class Node:
    def __init__(self, action=None, parent=None, prior_p=1.0):
        self.action = action
        self.parent = parent
        self.childs = {}

        self.V = 0 # action value
        self.N = 0 # visit count
        self.P = prior_p # prior probability of selecting that edge

    def select(self):
        # You should think carefully why use -V here
        return max(self.childs.values(), key=lambda c: -c.V + c.U())

    def expand(self, actions, probs):
        for i in range(len(actions)):
            a, p = actions[i], probs[i]
            n = Node(a, self, p)
            self.childs[a] = n
    
    def update(self, v):
        if self.N == 0 or v < self.V:
            self.V = v
        self.N += 1

    def back_propagate(self, v):
        self.update(v)
        if self.parent:
            self.parent.back_propagate(v)

    def U(self, c_puct=5.0):
        # c_puct -- a number in (0, inf) controlling the relative impact of
        #       values (Q) and prior probability (P) on this node's score
        #       it is a constant determining the level of exploration
        if self.parent:
            return c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return 0

    def __repr__(self):
        return "[A: %s, P: %.2f, V: %.2f, -V+U: %.2f, N: %d]" \
        % (self.action, self.P, self.V, -self.V + self.U(), self.N)

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

    def reset(self):
        self.rootnode = Node()

    def get_action(self, state, stochastically=False, show_node=False, verbose=False):
        for i in range(self.playout_itermax):
            self.playout(self.rootnode, state.clone())

        if show_node:
            if verbose: self.rootnode.show_node_tree()
            else: self.rootnode.show_children_nodes()

        action_probs = np.zeros((state.len, state.len))
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

    def update_with_action(self, action):
        if action in self.rootnode.childs:
            self.rootnode = self.rootnode.childs[action]
            self.rootnode.parent = None
        else:
            self.rootnode = Node()

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
        while node.childs != {}:
            node = node.select()
            state.move(node.action)

        # Rollout
        v, a, p = self.evaluate_state(state)
        node.expand(a, p)

        # Backpropagate
        node.back_propagate(v)


    def evaluate_state(self, state):
        L = state.len
        x = state.state.reshape((1, *state.state.shape))
        value, probs = self.f.pred(x)
        v = value[0, 0]
        a = state.action_space
        p = []
        probs = probs.reshape((L, L))
        for i, j in a:
            p.append(probs[i, j])
        p = np.array(p)
        if p.sum() > 0: p /= p.sum()
        return v, a, p

