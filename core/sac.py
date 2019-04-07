import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from core.utils import hard_update, soft_update


def initial_weights_(tensor):
    classname = tensor.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(tensor.weight, gain=1)
        nn.init.constant_(tensor.bias, 0)


class CriticV(nn.Module):
    def __init__(self, state_dim, layer_norm=False):
        super(CriticV, self).__init__()
        self.layer_norm = layer_norm

        l1_dim, l2_dim = 200, 300

        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        self.apply(initial_weights_)

    def forward(self, state):

        if not self.layer_norm:
            out = F.leaky_relu(self.l1(state))
            out = F.leaky_relu(self.l2(out))
            out = self.l3(out)
        else:
            out = F.leaky_relu(self.n1(self.l1(state)))
            out = F.leaky_relu(self.n2(self.l2(out)))
            out = self.l3(out)

        return out


class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim, layer_norm):
        super(CriticTD3, self).__init__()
        self.layer_norm = layer_norm

        l1_dim, l2_dim = 200, 300

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        # Q2 architecture
        self.L1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.L2 = nn.Linear(l1_dim, l2_dim)
        self.L3 = nn.Linear(l2_dim, 1)

        if layer_norm:
            self.N1 = nn.LayerNorm(l1_dim)
            self.N2 = nn.LayerNorm(l2_dim)

        self.apply(initial_weights_)

    def forward(self, state, action):

        if not self.layer_norm:
            # Q1 network output
            out1 = F.leaky_relu(self.l1(torch.cat([state, action], 1)))
            out1 = F.leaky_relu(self.l2(out1))
            out1 = self.l3(out1)
            # Q2 network output
            out2 = F.leaky_relu(self.L1(torch.cat([state, action], 1)))
            out2 = F.leaky_relu(self.L2(out2))
            out2 = self.L3(out2)

        else:
            # use layer normalization
            out1 = F.leaky_relu(self.n1(self.l1(torch.cat([state, action], 1))))
            out1 = F.leaky_relu(self.n2(self.l2(out1)))
            out1 = self.l3(out1)

            out2 = F.leaky_relu(self.N1(self.L1(torch.cat([state, action], 1))))
            out2 = F.leaky_relu(self.N2(self.L2(out2)))
            out2 = self.L3(out2)

        return out1, out2


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GaussianPolicy, self).__init__()

        # hidden layer dim
        l1_dim, l2_dim = 128, 128

        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l_mean = nn.Linear(l2_dim, action_dim)
        self.l_log_std = nn.Linear(l2_dim, action_dim)

        # Init
        self.apply(initial_weights_)

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        mean = self.l_mean(out)
        log_std = self.l_log_std(out).clamp(min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std


FloatTensor = torch.FloatTensor


class SAC(object):

    def __init__(self, args):

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.action_limit = args.action_limit

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        # Entropy regularization coefficient
        self.alpha = args.alpha

        # Whether use layer normalization in policy/value networks
        self.use_norm = args.use_norm
        # parameter for update policy/value networks
        self.tau = args.tau

        self._init_nets()

    def _init_nets(self):
        # initial one policy(P) net, two q(Q) nets, one v(V) net and one v_target_net(V_target)
        self.P = GaussianPolicy(self.state_dim, self.action_dim)
        self.Q = CriticTD3(self.state_dim, self.action_dim, self.use_norm)
        self.V = CriticV(self.state_dim, self.use_norm)
        self.V_target = CriticV(self.state_dim, self.use_norm)

        # initial optim
        self.p_optim = Adam(self.P.parameters(), lr=self.actor_lr)
        self.q_optim = Adam(self.Q.parameters(), lr=self.critic_lr)
        self.v_optim = Adam(self.V.parameters(), lr=self.critic_lr)

        # initial loss
        self.loss = nn.MSELoss()

        hard_update(self.V_target, self.V)

    def update(self, batch):
        with torch.no_grad():
            state_batch = batch['states']
            action_batch = batch['actions']
            next_state_batch = batch['next_states']
            reward_batch = batch['rewards']
            done_batch = batch['dones']
        # Compute q_target
        q1, q2 = self.Q(state_batch, action_batch)
        v_next_target = self.V_target(next_state_batch)
        q_target = reward_batch + self.gamma * (1 - done_batch.float()) * v_next_target.detach()

        # Compute v_target
        sample_actions, log_prob, _, mean, log_std = self.P.sample(state_batch)
        q1_new, q2_new = self.Q(state_batch, sample_actions)
        min_q_new = torch.min(q1_new, q2_new)
        v_target = min_q_new - (self.alpha * log_prob)
        v = self.V(state_batch)

        # q network update
        self.q_optim.zero_grad()
        q_loss = self.loss(q1, q_target) + self.loss(q2, q_target)
        q_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.Q.parameters(), 10)
        self.q_optim.step()

        # v network update
        self.v_optim.zero_grad()
        v_loss = self.loss(v, v_target.detach())
        v_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.V.parameters(), 10)
        self.v_optim.step()

        """
        Reparameterization trick is used to get a low variance estimator
        """
        # policy network update
        self.p_optim.zero_grad()
        mean_loss = 0.001 * mean.pow(2).mean()
        std_loss = 0.001 * log_std.pow(2).mean()
        p_loss = (self.alpha * log_prob - min_q_new).mean() + mean_loss + std_loss
        p_loss.backward()
        nn.utils.clip_grad_norm_(self.P.parameters(), 10)
        self.p_optim.step()

        soft_update(self.V_target, self.V, self.tau)
