import numpy as np
import torch
import gym
import tqdm

GAMMA = 0.995
ALPHA = 0.0001
N_STEPS = 1_000
N_EPOCHS = 10_000
BATCH_SIZE = 128
TAU = 0.9
SOFT_UPDATE = 100
EPSILON = 0.5
MAX_SIZE = 100_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EGreedyLinearStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        epsilon = 1 - self.t / self.decay_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * \
            epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))

        self.epsilon = self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)


class ReplayBuffer():
    def __init__(self,
                 max_size=MAX_SIZE,
                 batch_size=BATCH_SIZE):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
            np.vstack(self.as_mem[idxs]), \
            np.vstack(self.rs_mem[idxs]), \
            np.vstack(self.ps_mem[idxs]), \
            np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size


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


def fit_dqn(states, actions, rewards, next_states, dones, gamma, q_net, target_q_net, optimizer):
    max_a_q_sp = target_q_net(next_states).detach().max(1)[0].unsqueeze(1)
    target_q_s = rewards + gamma * max_a_q_sp * (1 - dones)
    q_sa = q_net(states).gather(1, actions)

    value_loss = torch.nn.functional.smooth_l1_loss(q_sa, target_q_s)
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()


def soft_update(target_net, online_net, tau=TAU):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data)


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
    print('Mean Reward after {} episodes = {}'.format(num_episodes, avg_reward))
    return avg_reward


def main():
    env = gym.make('Acrobot-v1')
    env._max_episode_steps = N_STEPS

    # Environment parameters
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    q_net = QNetwork(input_dim, output_dim)
    target_q_net = QNetwork(input_dim, output_dim)
    soft_update(q_net, target_q_net)
    strategy = EGreedyLinearStrategy()

    buffer = ReplayBuffer()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=ALPHA)

    for episode in tqdm.tqdm(range(N_EPOCHS+1)):
        state = env.reset()
        done = False

        while not done:
            action = strategy.select_action(q_net, state)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, float(done))

            buffer.store(experience)

            if len(buffer) >= 5*BATCH_SIZE:
                for i in range(10):
                    batched_experience = buffer.sample(BATCH_SIZE)
                    batched_experience = q_net.load(batched_experience)
                    fit_dqn(*batched_experience,
                            GAMMA, q_net, target_q_net, optimizer)
            state = next_state

        if episode % SOFT_UPDATE == 0:
            soft_update(target_q_net, q_net)

        if episode % 1000 == 0:
            eval_env = gym.make('Acrobot-v1')
            eval_env._max_episode_steps = N_STEPS
            evaluate(q_net, eval_env)

            eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
