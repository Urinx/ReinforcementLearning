import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym
from gym.spaces import Discrete, Box
import time
import random
import scipy.signal

def logger_print(logger, key, with_min_and_max=False):
    if with_min_and_max:
        print(f'{key+":":13s} {np.mean(logger[key]):.4f}\t{np.min(logger[key]):.4f}(min) {np.max(logger[key]):.4f}(max) {np.std(logger[key]):.4f}(std)')
    else:
        print(f'{key+":":13s} {np.mean(logger[key]):.4f}')

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

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

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x: [x0, x1, x2]

    output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size # buffer has to have room so you can store
        i = self.ptr
        self.obs_buf[i] = obs
        self.act_buf[i] = act
        self.rew_buf[i] = rew
        self.val_buf[i] = val
        self.logp_buf[i] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

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

        self.mlp = nn.Sequential(*net)

    def forward(self, x):
        return self.mlp(x)

class MLP_Categorical_Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, output_activation=None):
        super().__init__()

        self.mlp = MLP([obs_dim] + hidden_sizes + [act_dim], activation, output_activation)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.mlp(x)
        p = self.softmax(x)
        dist = Categorical(p)
        a = dist.sample()
        log_p = dist.log_prob(a)
        return a.item(), log_p

class MLP_Gaussian_Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, output_activation=None):
        super().__init__()

        self.mlp = MLP([obs_dim] + hidden_sizes + [act_dim], activation, output_activation)
        self.pi = torch.tensor(np.pi, dtype=torch.float)

    def forward(self, x, a=None):
        mu = self.mlp(x)
        log_std = -0.5 * torch.ones(mu.shape[-1], dtype=torch.float)
        std = torch.exp(log_std)
        if not self.training:
            a = mu + torch.randn(mu.shape) * std
        # gaussian likelihood
        pre_sum = -0.5 * ( ((a-mu) / (torch.exp(log_std) + 1e-8))**2 + 2*log_std + torch.log(2*self.pi) )
        logp = pre_sum.sum(dim=-1)
        return a, logp

class Actor_Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, output_activation=None, action_space=None):
        super().__init__()

        if isinstance(action_space, Box):
            policy = MLP_Gaussian_Policy
        elif isinstance(action_space, Discrete):
            policy = MLP_Categorical_Policy

        self.actor = policy(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        self.critic = MLP([obs_dim] + hidden_sizes + [1], activation, output_activation)

    def forward(self, x, a=None):
        v = self.critic(x)
        if self.training:
            _, logp = self.actor(x, a)
            return logp, v
        else:
            a, logp = self.actor(x)
            return a, logp, v

"""
Proximal Policy Optimization (by clipping), with early stopping based on approximate KL
"""
def train(
        env_name,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        target_kl=0.01,
        save_freq=10
        ):
    """

    Args:
        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    print(locals())

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Model
    actor_critic = Actor_Critic(obs_dim, act_dim, **ac_kwargs)
    print(actor_critic)
    print(f'\nNumber of parameters: {get_parameter_number(actor_critic)}\n')
    actor_critic.apply(weight_init)
    actor_optimizer = torch.optim.Adam(actor_critic.actor.parameters(), lr=pi_lr)
    critic_optimizer = torch.optim.Adam(actor_critic.critic.parameters(), lr=vf_lr)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    max_avg_ret = -np.inf
    for epoch in range(epochs):

        logger = {
            'VVals': [],
            'EpRet': [],
            'EpLen': [],
            'StopIter': [],
            'LossPi': [],
            'LossV': [],
            'KL': [],
            'Entropy': [],
            'ClipFrac': [],
            'DeltaLossPi': [],
            'DeltaLossV': []
        }

        actor_critic.eval()
        with torch.no_grad():
            for t in range(local_steps_per_epoch):
                a, logp, v = actor_critic(torch.tensor(o, dtype=torch.float))
                # breakpoint()

                # save and log
                buf.store(o, a, r, v, logp)
                logger['VVals'].append(v)

                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal or (t==local_steps_per_epoch-1):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else actor_critic(torch.tensor(o, dtype=torch.float))[-1].item()
                    buf.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger['EpRet'].append(ep_ret)
                        logger['EpLen'].append(ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        
        # Perform PPO update!
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()   
        obs = torch.tensor(obs_buf, dtype=torch.float)
        acts = torch.tensor(act_buf, dtype=torch.float)
        logp_old = torch.tensor(logp_buf, dtype=torch.float)
        adv = torch.tensor(adv_buf, dtype=torch.float)
        ret = torch.tensor(ret_buf, dtype=torch.float)

        actor_critic.train()
        with torch.no_grad():
            logp, v = actor_critic(obs, acts)
            ratio = torch.exp(logp - logp_old) # pi(a|s) / pi_old(a|s)
            min_adv = torch.where(adv>0, (1+clip_ratio)*adv, (1-clip_ratio)*adv)

            pi_l_old= - torch.min(ratio * adv, min_adv).mean()
            v_l_old = ((ret - v)**2).mean()
            ent = (-logp).mean() # a sample estimate for entropy, also easy to compute

        # Training
        for i in range(train_pi_iters):
            _, logp = actor_critic.actor(obs, acts)
            ratio = torch.exp(logp - logp_old) # pi(a|s) / pi_old(a|s)
            min_adv = torch.where(adv>0, (1+clip_ratio)*adv, (1-clip_ratio)*adv)
            pi_loss = - torch.min(ratio * adv, min_adv).mean()
            kl = (logp_old - logp).mean() # a sample estimate for KL-divergence, easy to compute

            actor_optimizer.zero_grad()
            pi_loss.backward()
            actor_optimizer.step()

            if kl > 1.5 * target_kl:
                # print('Early stopping at step %d due to reaching max kl.'%i)
                break

        logger['StopIter'].append(i)

        for _ in range(train_v_iters):
            v = actor_critic.critic(obs)
            v_loss = ((ret - v)**2).mean()

            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()

        # Log changes from update
        with torch.no_grad():
            logp, v = actor_critic(obs, acts)
            ratio = torch.exp(logp - logp_old) # pi(a|s) / pi_old(a|s)
            min_adv = torch.where(adv>0, (1+clip_ratio)*adv, (1-clip_ratio)*adv)
            pi_l_new= - torch.min(ratio * adv, min_adv).mean()
            v_l_new = ((ret - v)**2).mean()
            kl = (logp_old - logp).mean()
            clipped = np.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
            cf = clipped.float().mean()

            logger['LossPi'].append(pi_l_new)
            logger['LossV'].append(v_l_new)
            logger['KL'].append(kl)
            logger['Entropy'].append(ent)
            logger['ClipFrac'].append(cf)
            logger['DeltaLossPi'].append(pi_l_new - pi_l_old)
            logger['DeltaLossV'].append(v_l_new - v_l_old)

        # Log info about epoch
        print('-'*40)
        print(f'Epoch: {epoch}')
        print(f'TotalEnvInteracts: {(epoch+1)*steps_per_epoch}')
        logger_print(logger, 'EpRet', True)
        logger_print(logger, 'EpLen')
        logger_print(logger, 'VVals', True)
        logger_print(logger, 'LossPi')
        logger_print(logger, 'LossV')
        logger_print(logger, 'DeltaLossPi')
        logger_print(logger, 'DeltaLossV')
        logger_print(logger, 'Entropy')
        logger_print(logger, 'KL')
        logger_print(logger, 'ClipFrac')
        logger_print(logger, 'StopIter')
        print(f'Time: {time.time()-start_time:.4f}s')
        print('-'*40+'\n')

        # Save model
        if np.mean(logger['EpRet']) > max_avg_ret:
            max_avg_ret = np.mean(logger['EpRet'])
            torch.save(actor_critic.state_dict(), 'PPO_{}.pth'.format(env_name))

    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    train(
        args.env,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, 
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs
        )