import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability

from Utilities import ReplayBuffer
from CustomUtilities import print_graph
import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras

np.random.seed(42)
env_ = gym.make('CartPole-v1')


class DeepQLearningTensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape,
            alpha=1 / 1_000,
            gamma=99 / 100,
            epsilon_decay=9_999 / 10_000,
            batch_size=2 ** 6
    ):
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.n
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=2_000)
        self.main_nn = self.build_nn(hidden_shape)
        self.target_nn = keras.models.clone_model(model=self.main_nn)
        self.target_nn_hard_update()
        self.main_nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=alpha),
        )
        self.loss_function = keras.losses.MeanSquaredError()

    def build_nn(self, hidden_shape):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='leaky_relu')(layer)
        output_layer = keras.layers.Dense(units=self.n_a, activation='linear', dtype=tf.float64)(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    def target_nn_hard_update(self):
        self.target_nn.set_weights(self.main_nn.get_weights())

    #  hardcoded to work only for single state, add axis=1 on np.argmax for multiple states (returns np.ndarray)
    def choose_action(self, s: np.ndarray):
        s = tf.convert_to_tensor([s])
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.main_nn(s))

    #  hardcoded to work only for single state
    def choose_action_softmax(self, s: np.ndarray):
        s = tf.convert_to_tensor([s])
        state_action_value = self.main_nn(s)[0]
        # print(state_action_value)
        exp_actions = np.zeros(self.n_a)
        for a in range(self.n_a):
            exp_actions[a] = np.exp(state_action_value[a])
        softmax_values = exp_actions / np.sum(exp_actions)
        softmax_values = np.round(softmax_values, 5)  # only works for n_a = 2
        return np.random.choice(np.arange(self.n_a), p=softmax_values)

    #  hardcoded to work only for single state
    def choose_action_weighted_softmax(self, s: np.ndarray, weight):
        s = tf.convert_to_tensor([s])
        state_action_value = self.main_nn(s)[0]
        exp_actions = np.zeros(self.n_a)
        for a in range(self.n_a):
            exp_actions[a] = np.exp(state_action_value[a] * weight)
        softmax_values = exp_actions / np.sum(exp_actions)
        return np.random.choice(np.arange(self.n_a), p=softmax_values)

    def store_transition(self, s, a, r, s_, d):
        self.buffer.remember(s, a, r, s_, d)

    def train(self):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            batch_size=self.batch_size,
            cleared=False,
            randomized=True
        )
        states = tf.convert_to_tensor(states)
        states_ = tf.convert_to_tensor(states_)
        with tf.GradientTape() as tape:
            y_pred = self.main_nn(states)
            markov_pred = rewards + self.gamma * np.max(self.target_nn(states_), axis=1) * (1 - dones)
            y_hat = np.copy(y_pred)
            y_hat[np.arange(y_pred.shape[0]), actions] = markov_pred
            y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float32)
            loss = self.loss_function(y_pred, y_hat)
            # loss = tf.reduce_mean((y_hat - y_pred) ** 2)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.main_nn.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
        if self.epsilon > 1 / 100:
            self.epsilon = self.epsilon * self.epsilon_decay

    def fit(self, n_episodes=5_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes + 1):
            score = 0
            s = self.env.reset()[0]
            for i in range(self.env._max_episode_steps):
                # a = self.choose_action(s)
                a = self.choose_action_weighted_softmax(s, ep / (n_episodes+1))
                # a = self.choose_action_softmax(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                r = 0 if d == 0 else -100
                self.store_transition(s, a, r, s_, int(d))
                self.train()
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
                    self.main_nn.save(filepath='../data/dql_nn_1/main_nn')
                    print('model saved.')
                return self
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                # print('Ep %d | Epsilon: %.3f | Avg score: %.3f' % (ep, self.epsilon, avg_scores[-1]))
                print('Ep %d | weight: %.3f | Avg score: %.3f' % (ep, (ep / (n_episodes+1)), avg_scores[-1]))
                # print('Ep %d | Avg score: %.3f' % (ep, avg_scores[-1]))
                self.target_nn_hard_update()
            if ep % 500 == 0 and graph:
                print_graph(scores, avg_scores, 'scores', 'avg scores', 'DQL ep %d' % ep)
        return self

    def load_model(self, filepath='../data/dql_nn_1/main_nn'):
        self.main_nn = keras.models.load_model(filepath)

    def test_batch(self):
        for _ in range(5):
            s = self.env.reset()[0]
            a = self.env.action_space.sample()
            s_, r, d, t, _ = self.env.step(a)
            self.buffer.remember(s, a, r, s_, d)
        return self.buffer.get_buffer(batch_size=self.batch_size)

    def test_model(self, n_episodes=5):
        self.epsilon = 0
        self.load_model()
        for ep in range(1, n_episodes):
            score = 0
            s = self.env.reset()[0]
            while True:
                a = self.choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                if d or t:
                    break
                s = s_
            print('Episode %d | Score: %.3f' % (ep, score))


# agent = DeepQLearningTensorflow(env_, [16, 16, 32, 32])
# agent.fit(n_episodes=1_500)


class PolicyGradientMethodTensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape,
            alpha=1 / 1_000,
            gamma=99 / 100
    ):
        self.env = env
        self.n_s, self.n_a = env.observation_space.shape[0], env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer(max_size=5_000)
        self.nn = self._build_nn(hidden_shape)
        self.nn.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))

    def _build_nn(self, hidden_shape):
        def custom_softmax(x):
            return tf.exp(x) / tf.reshape(tf.reduce_sum(tf.exp(x), axis=1), [-1, 1])

        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=self.n_a, activation='softmax')(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    def _custom_policy_gradient_loss(self, states, actions, norm_returns):
        probabilities = self.nn(states)
        distribution = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
        log_probability_of_actions = distribution.log_prob(value=actions)
        loss = - (log_probability_of_actions * norm_returns)
        return loss

    #  hardcoded to work with a single state
    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        distribution = tfp.distributions.Categorical(probs=self.nn(s), dtype=tf.float32)
        return int(distribution.sample().numpy())

    def _store_transition(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def load_model(self, filepath='../data/pgm_nn_1/nn'):
        self.nn = keras.models.load_model(filepath)

    def _train(self):
        states, actions, rewards, _, _ = self.buffer.get_buffer(
            batch_size=self.buffer.get_buffer_size(),
            randomized=False,
            cleared=True
        )
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns = returns[::-1]
        normalized_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-16)
        with tf.GradientTape() as tape:
            # probabilities = self.nn(states)
            # distribution = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
            # log_probability_of_actions = distribution.log_prob(value=actions)
            # loss = - (log_probability_of_actions * normalized_returns)
            loss = self._custom_policy_gradient_loss(states, actions, normalized_returns)
        grads = tape.gradient(loss, self.nn.trainable_variables)
        self.nn.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))

    def fit(self, n_episodes=5_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes + 1):
            s = self.env.reset()[0]
            score = 0
            for i in range(self.env._max_episode_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                # r = 0 if d == 0 else -100
                self._store_transition(s, a, r)
                if d or t:
                    if i >= self.env._max_episode_steps - 1:
                        counter += 1
                    else:
                        counter = 0
                    break
                s = s_
            self._train()
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if counter >= 5:
                print('Environment %s solved at episode %d' % (self.env.unwrapped.spec.id, ep))
                if save_model:
                    self.nn.save(filepath='../data/pgm_nn_1/main_nn')
                    print('model saved.')
            if ep % 10 == 0:
                print('Episode %d | avg score: %.3f' % (ep, avg_scores[-1]))
            if ep % 200 == 0 and graph:
                print_graph(scores, avg_scores, 'scores', 'avg scores', 'PGM episode %d' % ep)
        return self

    def test_model(self, n_episodes=5):
        self.load_model()
        for ep in range(1, n_episodes):
            score = 0
            s = self.env.reset()
            while True:
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                if d or t:
                    break
                s = s_
            print('Episode %d | score: %.3f' % (ep, score))
        return self


class PolicyGradientActorCriticTensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape_1,
            hidden_shape_2,
            alpha=1 / 1_000,
            beta=1 / 500,
            gamma=99 / 100
    ):
        self.env = env
        self.n_s, self.n_a = env.observation_space.shape[0], env.action_space.n
        self.gamma = gamma
        self.buffer = ReplayBuffer(max_size=5_000)
        self.actor_nn = self._build_actor(hidden_shape_1)
        self.critic_nn = self._build_critic(hidden_shape_2)
        self.actor_nn.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.critic_nn.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.critic_loss_function = keras.losses.MeanSquaredError()

    def _build_actor(self, hidden_shape):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=self.n_a, activation='softmax')(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    #  outputting state value function
    def _build_critic(self, hidden_shape):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        distribution = tfp.distributions.Categorical(probs=self.actor_nn(s), dtype=tf.float32)
        return int(distribution.sample().numpy())

    def _store_transition(self, s, a, r):
        self.buffer.remember(s, a, r, None, None)

    def _load_model(self, filepath='../data/actor_critic_nn_1/actor_nn'):
        self.actor_nn = keras.models.load_model(filepath)

    def _train(self):

        states, actions, rewards, _, _ = self.buffer.get_buffer(
            batch_size=self.buffer.get_buffer_size(),
            randomized=False,
            cleared=True
        )
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns = returns[::-1]

        with tf.GradientTape() as actor_tape:

            probabilities = self.actor_nn(states)
            distribution = tfp.distributions.Categorical(probs=probabilities, dtype=tf.float32)
            actions_log_probability = distribution.log_prob(actions)

            state_values = self.critic_nn(states)
            advantages = np.reshape(returns, [-1, 1]) - state_values

            actor_loss = - tf.reduce_sum(actions_log_probability * advantages)

        actor_grads = actor_tape.gradient(actor_loss, self.actor_nn.trainable_variables)
        self.actor_nn.optimizer.apply_gradients(zip(actor_grads, self.actor_nn.trainable_variables))

        with tf.GradientTape() as critic_tape:

            state_values = self.critic_nn(states)
            critic_loss = self.critic_loss_function(returns, state_values)

        critic_grads = critic_tape.gradient(critic_loss, self.critic_nn.trainable_variables)
        self.critic_nn.optimizer.apply_gradients(zip(critic_grads, self.critic_nn.trainable_variables))

    def fit(self, n_episodes=5_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes + 1):
            s = self.env.reset()[0]
            score = 0
            for i in range(self.env._max_episode_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                score += r
                # r = 0 if d == 0 else -100
                self._store_transition(s, a, r)
                if d or t:
                    if i >= self.env._max_episode_steps - 1:
                        counter += 1
                    else:
                        counter = 0
                    break
                s = s_
            self._train()
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if counter >= 5:
                print('Environment %s solved at episode %d' % (self.env.unwrapped.spec.id, ep))
                if save_model:
                    self.actor_nn.save(filepath='../data/actor_critic_nn_1/actor_nn')
                    print('model saved.')
            if ep % 10 == 0:
                print('Episode %d | avg score: %.3f' % (ep, avg_scores[-1]))
            if ep % 200 == 0 and graph:
                print_graph(scores, avg_scores, 'scores', 'avg scores', 'PGM episode %d' % ep)
        return self


# agent = PolicyGradientActorCriticTensorflow(env_, [16, 16, 32, 32], [16, 32, 64])
# agent.fit(graph=False)

# agent = PolicyGradientMethodTensorflow(env_, [16, 16, 32, 32])
# agent.fit()

# agent = DeepQLearningTensorflow(env_, [16, 16, 32, 32])
# agent.fit(graph=False, save_model=False)
# agent.test_model()
