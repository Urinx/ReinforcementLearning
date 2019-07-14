import torch
import torch.nn as nn
import numpy as np
import gym
import time

def logger_print(logger, key, with_min_and_max=False):
    if with_min_and_max:
        print(f'{key+":":13s} {np.mean(logger[key]):.4f} {np.min(logger[key]):.4f}(min) {np.max(logger[key]):.4f}(max) {np.std(logger[key]):.4f}(std)')
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

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

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


class Actor_Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=nn.Tanh, action_space=None):
        super().__init__()

        self.actor = MLP([obs_dim] + hidden_sizes + [act_dim], activation, output_activation)
        self.critic = MLP([obs_dim + act_dim] + hidden_sizes + [1], activation, None)

"""
Deep Deterministic Policy Gradient (DDPG)
"""
def ddpg(
    env_name,
    ac_kwargs=dict(),
    seed=0, 
    steps_per_epoch=5000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    act_noise=0.1,
    max_ep_len=1000
    ):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

    """
    print(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make(env_name)
    test_env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Model
    main_ac = Actor_Critic(obs_dim, act_dim, **ac_kwargs)
    target_ac = Actor_Critic(obs_dim, act_dim, **ac_kwargs)
    print(main_ac)
    print(f'\nNumber of parameters: {get_parameter_number(main_ac)}\n')
    main_ac.apply(weight_init)

    pi_optimizer = torch.optim.Adam(main_ac.actor.parameters(), lr=pi_lr)
    q_optimizer = torch.optim.Adam(main_ac.critic.parameters(), lr=q_lr)
    mse_loss = nn.MSELoss()

    # copy main_ac nn parameters to target_ac
    for v_targ, v_main in zip(target_ac.parameters(), main_ac.parameters()):
        v_targ.data.copy_(v_main.data)

    # Main loop: collect experience in env and update/log each epoch
    t = 0
    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    max_avg_ret = -np.inf

    for epoch in range(epochs):
        logger = {
            'LossQ': [],
            'QVals': [],
            'LossPi': [],
            'EpRet': [],
            'EpLen': [],
            'TestEpRet': [],
            'TestEpLen': []
        }

        for _ in range(steps_per_epoch):
            """
            Until start_steps have elapsed, randomly sample actions
            from a uniform distribution for better exploration. Afterwards, 
            use the learned policy (with some noise, via act_noise). 
            """
            if t > start_steps:
                with torch.no_grad():
                    pi = act_limit * main_ac.actor(torch.tensor(o, dtype=torch.float))
                    pi = pi.numpy() + act_noise * np.random.randn(act_dim)
                    a = np.clip(pi, -act_limit, act_limit)
            else:
                a = env.action_space.sample()

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            if d or (ep_len == max_ep_len):
                """
                Perform all DDPG updates at the end of the trajectory,
                in accordance with tuning done by TD3 paper authors.
                """
                for _ in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    obs1 = torch.tensor(batch['obs1'], dtype=torch.float)
                    obs2 = torch.tensor(batch['obs2'], dtype=torch.float)
                    acts = torch.tensor(batch['acts'], dtype=torch.float)
                    rews = torch.tensor(batch['rews'], dtype=torch.float).unsqueeze(1)
                    done = torch.tensor(batch['done'], dtype=torch.float).unsqueeze(1)

                    # Q-learning update
                    q = main_ac.critic(torch.cat([obs1, acts], dim=-1))
                    pi_targ = act_limit * target_ac.actor(obs2)
                    q_pi_targ = target_ac.critic(torch.cat([obs2, pi_targ], dim=-1))
                    backup = rews + gamma * (1 - done) * q_pi_targ
                    q_loss = mse_loss(q, backup.detach())

                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()
                    logger['LossQ'].append(q_loss.item())
                    logger['QVals'] += q.squeeze().tolist()

                    # Policy update
                    pi = act_limit * main_ac.actor(obs1)
                    q_pi = main_ac.critic(torch.cat([obs1, pi], dim=-1))
                    pi_loss = -q_pi.mean()

                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    pi_optimizer.step()

                    logger['LossPi'].append(pi_loss.item())

                    # Target update
                    for v_targ, v_main in zip(target_ac.parameters(), main_ac.parameters()):
                        v_targ.data.copy_(polyak * v_targ.data + (1 - polyak) * v_main.data)

                logger['EpRet'].append(ep_ret)
                logger['EpLen'].append(ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            t += 1

        # Test the performance of the deterministic version of the agent.
        with torch.no_grad():
            for _ in range(10):
                ob, ret, done, test_ep_ret, test_ep_len = test_env.reset(), 0, False, 0, 0
                while not(done or (test_ep_len == max_ep_len)):
                    # Take deterministic actions at test time withour noise
                    pi = act_limit * main_ac.actor(torch.tensor(ob, dtype=torch.float))
                    act = np.clip(pi, -act_limit, act_limit)
                    ob, ret, done, _ = test_env.step(act)
                    test_ep_ret += ret
                    test_ep_len += 1
                logger['TestEpRet'].append(test_ep_ret)
                logger['TestEpLen'].append(test_ep_len)

        # Log info about epoch
        print('-'*40)
        print(f'Epoch: {epoch}')
        print(f'TotalEnvInteracts: {t}')
        logger_print(logger, 'EpRet', True)
        logger_print(logger, 'EpLen')
        logger_print(logger, 'TestEpRet', True)
        logger_print(logger, 'TestEpLen')
        logger_print(logger, 'QVals', True)
        logger_print(logger, 'LossPi')
        logger_print(logger, 'LossQ')
        print(f'Time: {time.time()-start_time:.4f}s')
        print('-'*40+'\n')

        # Save model
        if np.mean(logger['EpRet']) > max_avg_ret:
            max_avg_ret = np.mean(logger['EpRet'])
            torch.save(main_ac.state_dict(), 'DDPG_{}.pth'.format(env_name))

    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    ddpg(
        args.env,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs
        )