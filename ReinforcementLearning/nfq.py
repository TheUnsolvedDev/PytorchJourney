import numpy as np
import torch
import gym
import random
import tqdm

GAMMA = 0.995
ALPHA = 0.001
N_STEPS = 1000
N_EPOCHS = 10000
BATCH_SIZE = 1024
TAU = 0.01
SOFT_UPDATE = 25
EPSILON = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EGreedyStrategy():
    def __init__(self, epsilon=EPSILON):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)


class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.to(device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self._format(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        new_states = torch.from_numpy(new_states).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        is_terminals = torch.from_numpy(is_terminals).float().to(device)
        return states, actions, rewards, new_states, is_terminals


def fit_q_learning(states, actions, rewards, next_states, dones, gamma, q_net, optimizer):
    max_a_q_sp = q_net(next_states).detach().max(1)[
        0].unsqueeze(1)
    target_q_s = rewards + gamma*max_a_q_sp*(1-dones)
    q_sa = q_net(states).gather(1, actions)

    td_errors = q_sa - target_q_s
    value_loss = td_errors.pow(2).mul(0.5).mean()
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()


def evaluate(q_net, env, num_episodes=10):
    total_rewards = []
    strategy = GreedyStrategy()
    for episode in range(num_episodes):
        state = env.reset()
        # env.render()
        done = False
        episode_reward = 0

        while not done:
            action = strategy.select_action(q_net, state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print('Mean Reward after {} episodes = {}'.format(num_episodes,avg_reward))
    return avg_reward


def main():
    env = gym.make('CartPole-v1')
    env._max_episode_steps = N_STEPS

    # Environment parameters
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    q_net = QNetwork(input_dim, output_dim)
    strategy = EGreedyStrategy(EPSILON)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=ALPHA)
    experiences = []

    for episode in tqdm.tqdm(range(N_EPOCHS)):
        state = env.reset()
        done = False
        batches = []
        while not done:
            action = strategy.select_action(q_net, state)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, float(done))

            experiences.append(experience)

            if len(experiences) >= BATCH_SIZE:
                for _ in range(20):
                    sample = random.sample(experiences, BATCH_SIZE//16)
                    batched_experience = map(np.asarray, zip(*sample))
                    batches = [np.vstack(sars) for sars in batched_experience]
                    batched_experience = q_net.load(batches)
                    fit_q_learning(*batched_experience, GAMMA, q_net, optimizer)
                experiences.clear()
            state = next_state
            
        if episode % 1000 == 0:
            eval_env = gym.make('CartPole-v1')
            eval_env._max_episode_steps = N_STEPS
            evaluate(q_net, eval_env)

            eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
