import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym
from gym.spaces import Discrete, Box
import argparse
import random

seed = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    DEBUG = False
else:
    DEBUG = True

def weight_init(m):
    '''
    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh, output_activation=None):
        super().__init__()

        net = []
        for i in range(len(sizes)-1):
            net.append(nn.Linear(sizes[i], sizes[i+1]))
            if i == len(sizes) - 2:
                if output_activation is not None:
                    net.append(output_activation())
            else:
                net.append(activation())

        self.mlp = nn.Sequential(
            *net,
            nn.Softmax(dim=-1)
            )

    def forward(self, x):
        return self.mlp(x)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy = MLP(sizes=[obs_dim]+hidden_sizes+[n_acts])
    policy.apply(weight_init)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        policy.eval()
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            with torch.no_grad():
                act_probs = policy(torch.tensor(obs, dtype=torch.float))
                dist = Categorical(act_probs)
                act = dist.sample().item()
            
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        policy.train()
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts)
        batch_weights = torch.tensor(batch_weights)

        batch_act_probs = policy(batch_obs)
        dist = Categorical(batch_act_probs)
        log_probs = dist.log_prob(batch_acts)
        loss = (- log_probs * batch_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, batch_rets, batch_lens

    # training loop
    max_avg_ret = 0
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(f'epoch: {i:2d} loss: {batch_loss:.3f} episode average rewards: {np.mean(batch_rets):.3f} episode average len: {np.mean(batch_lens):.3f}')

        if np.mean(batch_rets) > max_avg_ret:
            max_avg_ret = np.mean(batch_rets)
            torch.save(policy.state_dict(), 'PG_{}.pth'.format(env_name))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr, epochs=args.epochs)