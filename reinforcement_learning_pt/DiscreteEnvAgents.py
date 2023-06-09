import gym
import numpy as np
from reinforcement_learning_tf.Utilities import ReplayBuffer
from CustomUtilities import print_graph
import torch
nn = torch.nn

np.random.seed(42)
env_ = gym.make('CartPole-v1')


class NN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.layer_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.layer_2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, X):
        X_1 = self.layer_1(X)
        Z_1 = torch.relu(X_1)
        out = self.layer_2(Z_1)
        #  do not apply softmax since crossentropy does it automatically in pytorch
        return out


class DeepQLearningPytorch:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape,
            alpha=1 / 1_000,
            gamma=99 / 100,
            epsilon_decay=9_999 / 10_000,
            batch_size=2 ** 6
    ):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.n
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=2_000)
        self.main_nn = self._build_nn(hidden_shape)
        self.target_nn = self._build_nn(hidden_shape)
        self._target_nn_hard_update()
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(params=self.main_nn.parameters(), lr=alpha)

    def _build_nn(self, hidden_shape):
        model = nn.Sequential()
        for i in range(len(hidden_shape)):
            module = nn.Linear(hidden_shape[i-1] if i > 0 else self.n_s, hidden_shape[i])
            model.add_module(name='l_%d' % (i+1), module=module)
            model.add_module(name='a_%d' % (i+1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(hidden_shape[-1], self.n_a))
        return model

    def _target_nn_hard_update(self):
        with torch.no_grad():
            self.target_nn.load_state_dict(self.main_nn.state_dict())

    #  hardcoded to work only for single state, add axis=1 on np.argmax for multiple states (returns np.ndarray)
    def _choose_action(self, s: np.ndarray):
        s = torch.from_numpy(s)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(torch.Tensor.numpy(self.main_nn(s).detach()))

    #  hardcoded to work only for single state
    def _choose_action_softmax(self, s: np.ndarray):
        s = torch.from_numpy(s)
        state_action_value = self.main_nn(s).detach()
        exp_actions = np.zeros(self.n_a)
        for a in range(self.n_a):
            exp_actions[a] = np.exp(state_action_value[a])
        softmax_values = exp_actions / np.sum(exp_actions)
        softmax_values = np.round(softmax_values, 5)  # only works for n_a = 2
        return np.random.choice(np.arange(self.n_a), p=softmax_values)

    #  hardcoded to work only for single state
    def _choose_action_weighted_softmax(self, s: np.ndarray, weight):
        s = torch.from_numpy(s)
        state_action_value = self.main_nn(s).detach()
        exp_actions = np.zeros(self.n_a)
        for a in range(self.n_a):
            exp_actions[a] = np.exp(state_action_value[a] * weight)
        softmax_values = exp_actions / np.sum(exp_actions)
        return np.random.choice(np.arange(self.n_a), p=softmax_values)

    def _store_transition(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def _train(self):

        def preprocess_action(acts):
            return torch.from_numpy(
                np.array([[np.int64(a)] for a in acts])
            )

        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            batch_size=self.batch_size,
            cleared=False,
            randomized=True
        )
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        states_ = torch.from_numpy(states_)
        dones = torch.from_numpy(dones)
        y_pred = torch.gather(input=self.main_nn(states), dim=1, index=preprocess_action(actions))
        y_pred = torch.squeeze(y_pred)
        with torch.no_grad():
            y_pred_next, _ = torch.max(self.target_nn(states_), dim=1)
        y_hat = rewards + self.gamma * y_pred_next * (1 - dones)
        loss = self.criterion(y_pred, y_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > 1 / 100:
            self.epsilon = self.epsilon * self.epsilon_decay

    def fit(self, n_episodes=5_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes + 1):
            score = 0
            s = self.env.reset()[0]
            for i in range(self.env._max_episode_steps):
                # a = self._choose_action_weighted_softmax(s, ep / (n_episodes+1))
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                r = 0 if d == 0 else -100
                self._store_transition(s, a, r, s_, int(d))
                self._train()
                if d or t:
                    if i >= self.env._max_episode_steps - 1:
                        counter += 1
                    else:
                        counter = 0
                    break
                s = s_
            if counter >= 5:
                print('Environment %s solved at episode %d' % (self.env.unwrapped.spec.id, ep))
                if save_model:
                    pass
                return self
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Ep %d | Epsilon: %.3f | Avg score: %.3f' % (ep, self.epsilon, avg_scores[-1]))
                # print('Ep %d | weight: %.3f | Avg score: %.3f' % (ep, (ep / (n_episodes+1)), avg_scores[-1]))
                # print('Ep %d | Avg score: %.3f' % (ep, avg_scores[-1]))
                self._target_nn_hard_update()
            if ep % 500 == 0 and graph:
                print_graph(scores, avg_scores, 'scores', 'avg scores', 'DQL ep %d' % ep)
        return self

    def _load_model(self, filepath):
        self.main_nn = torch.load(f=filepath)

    def test_batch(self):
        for _ in range(5):
            s = self.env.reset()[0]
            a = self.env.action_space.sample()
            s_, r, d, t, _ = self.env.step(a)
            self.buffer.remember(s, a, r, s_, d)
        return self.buffer.get_buffer(batch_size=self.batch_size)

    def test_model(self, n_episodes, load_path: str):
        self.epsilon = 0
        self._load_model(load_path)
        for ep in range(1, n_episodes):
            score = 0
            s = self.env.reset()[0]
            while True:
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                if d or t:
                    break
                s = s_
            print('Episode %d | Score: %.3f' % (ep, score))


# agent = DeepQLearningPytorch(env_, [16, 16, 32, 32])
# agent.fit()


class PolicyGradientMethodPytorch:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape,
            alpha = 1 / 1_000,
            gamma=99 / 100,
            batch_size=2 ** 6
    ):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=2_000)
        self.actor_nn = self._build_nn(hidden_shape)
        self.criterion = self._custom_loss
        self.optimizer = torch.optim.Adam(params=self.actor_nn.parameters(), lr=alpha)

    def _custom_loss(self, states, actions, norm_returns):
        states = torch.Tensor(states)
        actions = torch.Tensor(actions)
        probabilities = self.actor_nn(states)
        distribution = torch.distributions.Categorical(probs=probabilities)
        log_action_probas = distribution.log_prob(value=actions)
        losses = torch.sum(-log_action_probas * norm_returns)
        return losses

    def _build_nn(self, hidden_shape):
        model = nn.Sequential()
        for i in range(len(hidden_shape)):
            module = nn.Linear(hidden_shape[i - 1] if i > 0 else self.n_s, hidden_shape[i])
            model.add_module(name='l_%d' % (i + 1), module=module)
            model.add_module(name='a_%d' % (i + 1), module=nn.ReLU())
        model.add_module(name='l_out', module=nn.Linear(hidden_shape[-1], self.n_a))
        model.add_module(name='a_out', module=nn.Softmax(dim=-1))
        return model

    def _store_transition(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _choose_action(self, s):
        s = torch.Tensor(s)
        with torch.no_grad():
            distribution = torch.distributions.Categorical(probs=self.actor_nn(s))
        return distribution.sample().item()

    def _train(self):
        pass

    def fit(self, n_episodes=2_000, graph=True, save_model=False):
        pass

    def t_model(self, n_episodes=5):
        pass

    def t_losses(self):
        s = self.env.reset()[0]
        for _ in range(5):
            a = self.env.action_space.sample()
            s_, r, d, t, _ = self.env.step(a)
            self._store_transition(s, a, r)
            s = s_
        return self.buffer.get_buffer(batch_size=self.buffer.get_buffer_size(), randomized=False, cleared=True)


agent = PolicyGradientMethodPytorch(env_, [16, 32, 64])
nn = agent.actor_nn
# states, actions, rewards, _, _ = agent.t_losses()






















































































































