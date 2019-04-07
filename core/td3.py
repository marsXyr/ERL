import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import hard_update, soft_update, to_tensor


def initial_weights_(tensor):
    classname = tensor.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(tensor.weight, gain=1)
        nn.init.constant_(tensor.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(Actor, self).__init__()

        self.layer_norm = layer_norm
        # hidden layer dim
        l1_dim, l2_dim = 128, 128

        self.l1 = nn.Linear(state_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, action_dim)

        # use layer normalization
        if layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)

        # Init
        self.apply(initial_weights_)

    def forward(self, state):

        if not self.layer_norm:
            out = torch.tanh(self.l1(state))
            out = torch.tanh(self.l2(out))
            out = torch.tanh(self.l3(out))
        else:
            out = torch.tanh(self.n1(self.l1(state)))
            out = torch.tanh(self.n2(self.l2(out)))
            out = torch.tanh(self.l3(out))

        return out


class Critic_TD3(nn.Module):
    def __init__(self, state_dim, action_dim, layer_norm):
        super(Critic_TD3, self).__init__()
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


class TD3(object):

    def __init__(self, args):

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.action_limit = args.action_limit

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.gamma = args.gamma
        # Stddev for smoothing noise added to target policy
        self.target_noise = args.target_noise
        # Limit for absolute value of target policy
        self.noise_clip = args.noise_clip
        self.batch_size = args.batch_size

        # Whether use layer normalization in policy/value networks
        self.use_norm = args.use_norm
        # parameter for update policy/value networks
        self.tau = args.tau

        self._init_nets()

    def _init_nets(self):
        # initial actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, self.use_norm)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.use_norm)
        self.critic = Critic_TD3(self.state_dim, self.action_dim, self.use_norm)
        self.critic_target = Critic_TD3(self.state_dim, self.action_dim, self.use_norm)

        # initial optim
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        # initial loss
        self.loss = nn.MSELoss()

        # initial the actor target and critic target are the same as actor and critic
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)


    def update(self, batch, policy_update=False):
        with torch.no_grad():
            state_batch = batch['states']
            action_batch = batch['actions']
            next_state_batch = batch['next_states']
            reward_batch = batch['rewards']
            done_batch = batch['dones']

        # Target policy smoothing, by adding clipped noise to target actions
        noise = np.clip(np.random.normal(0, self.target_noise, size=(self.batch_size, self.action_dim)),
                        -self.noise_clip, self.noise_clip)
        next_action = self.actor_target(next_state_batch) + to_tensor(noise)
        next_action_clip = next_action.clamp(-self.action_limit, self.action_limit)
        with torch.no_grad():
            q1_next, q2_next = self.critic_target(next_state_batch, next_action_clip)
            min_q_next = torch.min(q1_next, q2_next)
            # compute q_target and two q predict
            q_target = reward_batch + self.gamma * (1 - done_batch.float()) * min_q_next
        q1_predict, q2_predict = self.critic(state_batch, action_batch)

        # critic update
        self.critic_optim.zero_grad()
        critic_loss = self.loss(q1_predict, q_target) + self.loss(q2_predict, q_target)
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        if policy_update:
            # Delayed policy update
            self.actor_optim.zero_grad()
            q1, _ = self.critic(state_batch, self.actor(state_batch))
            actor_loss = -q1.mean()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optim.step()

            # actor/critic network soft update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

