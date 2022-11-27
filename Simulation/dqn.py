import random
import datetime
import pickle
import torch
import numpy as np
import gc
import math
import multiprocessing as mp
from multiprocessing import set_start_method
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from ReplayMemory import ReplayMemory
from LoraEnvironment import LoraEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_space_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * 2),
            nn.ReLU(),
            nn.Linear(64 * 2, action_space_dim)
        )

    def forward(self, x):
        x = x.to(device)
        return self.linear(x)


def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
    # Sample the data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([s[1] for s in batch], dtype=torch.int64, device=device)
    rewards = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=device)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32,
                                         device=device)  # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1).cuda())

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
        target_net.eval()
        q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size, device=device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)  # Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

    return loss.item()


def plot_reward(plotting_rewards, num_iterations):

    plt.figure(figsize=(10, 8))
    plt.plot(range(0, num_iterations), plotting_rewards)
    plt.xlabel('Training episodes')
    plt.ylabel('Acc. Episodic Reward')
    plt.grid()
    plt.legend()
    plt.show()


def run(args):
    return LoraEnvironment(args[0], args[1], args[2]).run()


def main():
    set_start_method('spawn')

    state_space_dim = 6
    action_space_dim = 12

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # PARAMETERS
    gamma = 0.9  # gamma parameter for the long term reward
    replay_memory_capacity = 20000  # Replay memory capacity
    lr = 1e-3
    target_net_update_steps = 5  # Number of episodes to wait before updating the target network
    batch_size = 64  # Number of samples to take from the replay memory for each update
    bad_state_penalty = 0  # Penalty to the reward when we are in a bad state (in this case when the pole falls down)
    min_samples_for_training = 64  # Minimum samples in the replay memory to enable the training
    initial_value = 3
    num_iterations = 100

    replay_mem = ReplayMemory(replay_memory_capacity)

    policy_net = DQN(state_space_dim, action_space_dim).to(device)

    # Initialize the target network with the same weights of the policy network
    target_net = DQN(state_space_dim, action_space_dim).to(device)

    # This will copy the weights of the policy network to the target network
    target_net.load_state_dict(policy_net.state_dict())

    # The optimizer will update ONLY the parameters of the policy network
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    loss_fn = nn.SmoothL1Loss()

    plotting_rewards = []

    # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
    exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6)
    exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

    #Plot exploration profile
    # plt.figure(figsize=(12, 8))
    # plt.plot(exploration_profile)
    # plt.grid()
    # plt.xlabel('Iteration')
    # plt.ylabel('Exploration profile (Softmax temperature)')
    # plt.show()

    for episode_num, tau in enumerate(tqdm(exploration_profile)):

        #pool = mp.Pool(math.floor(mp.cpu_count()))

        score = 0

        # Run environment
        args = [[policy_net, tau, 2], [policy_net, tau, 3]]
        #r_list = pool.map(func=run, iterable=args)
        r_list = [run(args[0])]

        for _r in r_list:
            score_r, replay = _r
            replay_mem.extend(replay)
            score += score_r

        print(f"Length replay: {len(replay_mem)}")

        # Update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            print('Updating target network...')

            # This will copy the weights of the policy network to the target network
            target_net.load_state_dict(policy_net.state_dict())

        plotting_rewards.append(score)

        if len(replay_mem) > min_samples_for_training:
            for _ in range(3000):
                update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}")

    filename = datetime.datetime.now()

    torch.save(target_net, f'models/weights_{filename}.pt')

    with open(f'models/score_{filename}.pkl', 'wb') as f:
        pickle.dump(plotting_rewards, f)

    plot_reward(plotting_rewards, num_iterations)


if __name__ == '__main__':
    main()

