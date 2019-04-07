import numpy as np, os, time, sys, random
from core.neuro_evolutionary import SSNE
from core.utils import Tracker, NormalizedActions, OUNoise, to_tensor, to_numpy
import gym, torch
from core import replay_memory
from core import td3 as td3
import argparse


render = False
parser = argparse.ArgumentParser()
#parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
parser.add_argument('env', type=str, default='HalfCheetah-v2')
#env_tag = vars(parser.parse_args())['env']
env_tag = 'HalfCheetah-v2'


class Parameters:
    def __init__(self):

        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 4000000
        elif env_tag == 'Ant-v2': self.num_frames = 6000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 8000000
        else: self.num_frames = 1000000

        # Synchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2':
            self.synch_period = 1
        else:
            self.synch_period = 10

        #TD3 params
        self.use_norm = True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 7
        self.batch_size = 128
        self.buffer_size = int(1e6)
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.start_frames = 10000
        self.policy_decay = 2
        self.target_noise = 0.2
        self.noise_clip = 0.5
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        ###### NeuroEvolution Params ########
        #Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1


        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None    # Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_ETD3/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = SSNE(self.args)

        # Init population
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(td3.Actor(args.state_dim, args.action_dim, layer_norm=True))

        # Turn off gradients and put in eval mode
        for actor in self.pop: actor.eval()        # train = False

        # Init RL Agent
        self.rl_agent = td3.TD3(args)
        self.replay_buffer = replay_memory.ReplayBuffer(args.buffer_size, args.state_dim, args.action_dim)
        self.ounoise = OUNoise(args.action_dim)

        # Trackers
        self.num_games = 0
        self.num_frames = 0
        self.start_frames = args.start_frames
        self.gen_frames = None

    def evaluate(self, net, is_render, is_action_noise=False, store_transition=True):

        state, reward, done, ep_return = self.env.reset(), 0, False, 0

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if render and is_render: self.env.render()
            action = net.forward(to_tensor(state).unsqueeze(0))
            action = to_numpy(action.clamp(-1, 1))
            if is_action_noise: action += self.ounoise.add()

            next_state, reward, done, info = self.env.step(action)  #Simulate one step in environment
            ep_return += reward
            done_bool = float(done)
            if store_transition: self.replay_buffer.add((state, action, next_state, reward, done_bool))
            state = next_state
        if store_transition: self.num_games += 1

        return ep_return

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        self.gen_frames = 0

        ####################### EVOLUTION #####################
        all_fitness = []
        #Evaluate genomes/individuals
        for net in self.pop:
            fitness = 0.0
            for eval in range(self.args.num_evals): fitness += self.evaluate(net, is_render=False, is_action_noise=False)
            all_fitness.append(fitness/self.args.num_evals)

        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        #Validation test
        champ_index = all_fitness.index(max(all_fitness))
        test_score = 0.0
        for eval in range(5): test_score += self.evaluate(self.pop[champ_index], is_render=True, is_action_noise=False, store_transition=False)/5.0

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)


        ####################### DDPG #########################
        #DDPG Experience Collection
        self.evaluate(self.rl_agent.actor, is_render=False, is_action_noise=True) # Train

        #DDPG learning step
        if self.num_frames > self.start_frames:
            for _ in range(int(self.gen_frames*self.args.frac_frames_train)):
                batch = self.replay_buffer.sample(self.args.batch_size)
                # Delayed policy update
                policy_update = True if (self.num_frames % self.args.policy_decay == 0) else False
                self.rl_agent.update(batch, policy_update=policy_update)

            #Synch RL Agent to NE
            if self.num_games % self.args.synch_period == 0:
                self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
                self.evolver.rl_policy = worst_index
                print('Synch from RL --> Nevo')

        return best_train_fitness, test_score, elite_index

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters, ['etd3'], '_score.csv')  # Initiate tracker
    frame_tracker = Tracker(parameters, ['frame_etd3'], '_score.csv')  # Initiate tracker
    time_tracker = Tracker(parameters, ['time_etd3'], '_score.csv')

    #Create Env
    env = NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.action_limit = env.action_space.high[0]

    #Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    #Create Agent
    agent = Agent(parameters, env)
    print('Running', env_tag, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    next_save = 100; time_start = time.time()
    while agent.num_frames <= parameters.num_frames:
        best_train_fitness, etd3_score, elite_index = agent.train()
        print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness if best_train_fitness != None else None, ' Test_Score:','%.2f'%etd3_score if etd3_score != None else None, ' Avg:','%.2f'%tracker.all_tracker[0][1], 'ENV '+env_tag)
        print('RL Selection Rate: Elite/Selected/Discarded', '%.2f'%(agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']),
                                                             '%.2f' % (agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']),
                                                              '%.2f' % (agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']))
        print()
        tracker.update([etd3_score], agent.num_games)
        frame_tracker.update([etd3_score], agent.num_frames)
        time_tracker.update([etd3_score], time.time()-time_start)

        #Save Policy
        if agent.num_games > next_save:
            next_save += 100
            if elite_index != None: torch.save(agent.pop[elite_index].state_dict(), parameters.save_foldername + 'evo_net')
            print("Progress Saved")











