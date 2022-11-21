import random
import datetime
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from collections import deque, namedtuple
from LoraEnvironment import LoraEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        # Define a queue with maxlen "capacity"
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        batch_size = min(batch_size, len(self))

        # Randomly select "batch_size" samples and return the selection
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  # Return the number of samples currently stored in the memory


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


def choose_action_epsilon_greedy(net, state, epsilon):
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')

    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)  # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions (this list includes all the actions but the optimal one)
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly from non_optimal_actions
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action

    return action, net_out.cpu().numpy()


def choose_action_softmax(net, state, temperature):
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')

    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)

    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8)  # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out / temperature, dim=0).cpu().numpy()

    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])

    # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    action = np.random.choice(all_possible_actions, p=softmax_out)

    return action, net_out.cpu().numpy()


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


def main():

    # Create environment
    env = LoraEnvironment()
    env.seed(0)

    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    print(f"STATE SPACE SIZE: {state_space_dim}")
    print(f"ACTION SPACE SIZE: {action_space_dim}")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # PARAMETERS
    gamma = 0.99  # gamma parameter for the long term reward
    replay_memory_capacity = 1000  # Replay memory capacity
    lr = 1e-3
    target_net_update_steps = 5  # Number of episodes to wait before updating the target network
    batch_size = 64  # Number of samples to take from the replay memory for each update
    bad_state_penalty = 0  # Penalty to the reward when we are in a bad state (in this case when the pole falls down)
    min_samples_for_training = 64  # Minimum samples in the replay memory to enable the training
    initial_value = 5
    num_iterations = 300

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

    # Plot exploration profile
    # plt.figure(figsize=(12, 8))
    # plt.plot(exploration_profile)
    # plt.grid()
    # plt.xlabel('Iteration')
    # plt.ylabel('Exploration profile (Softmax temperature)')
    # plt.show()

    for episode_num, tau in enumerate(tqdm(exploration_profile)):

        # Reset the environment and get the initial state
        state = env.reset()
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False

        # Go on until the pole falls off
        while not done:

            # Choose the action following the policy
            action, q_values = choose_action_softmax(policy_net, state, temperature=tau)

            # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
            next_state, reward, done, info = env.step(action)

            # Update the final score (+1 for each step)
            score += reward

            # Apply penalty for bad state
            if done:  # if the pole has fallen down
                reward += bad_state_penalty
                next_state = None

            # Update the replay memory
            replay_mem.push(state, action, next_state, reward)

            # Update the network
            # we enable the training only if we have enough samples in the replay memory,
            # otherwise the training will use the same samples too often
            loss = 0
            if len(replay_mem) > min_samples_for_training:
                loss = update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

            print(action, next_state, reward, loss)

            # Set the current state for the next iteration
            state = next_state

        # Update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            print('Updating target network...')

            # This will copy the weights of the policy network to the target network
            target_net.load_state_dict(policy_net.state_dict())

        plotting_rewards.append(score)

        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}")  # Print the final score

    env.close()

    filename = datetime.datetime.now()

    torch.save(target_net, f'models/weights_{filename}.pt')

    with open(f'models/score_{filename}.pkl', 'wb') as f:
        pickle.dump(plotting_rewards, f)

    plot_reward(plotting_rewards, num_iterations)


if __name__ == '__main__':
    main()

