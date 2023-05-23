import gym
import matplotlib.pyplot as plt
import numpy as np
from Utilities import ReplayBuffer
from CustomUtilities import print_graph
import tensorflow as tf
keras = tf.keras


np.random.seed(42)
env_ = gym.make('CartPole-v1')


class DeepQLearningTensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape,
            alpha=1/1_000,
            gamma=99/100,
            epsilon_decay=9_999/10_000,
            batch_size=2**6
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
        output_layer = keras.layers.Dense(units=self.n_a, activation='linear')(layer)
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
            # loss = self.loss_function(y_pred, y_hat)
            loss = tf.reduce_mean((y_hat - y_pred) ** 2)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.main_nn.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
        if self.epsilon > 1/100:
            self.epsilon = self.epsilon * self.epsilon_decay

    def fit(self, n_episodes=5_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            for i in range(self.env._max_episode_steps):
                a = self.choose_action(s)
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
                return self
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            if ep % 10 == 0:
                print('Ep %d | Epsilon: %.3f | Avg score: %.3f' % (ep, self.epsilon, avg_scores[-1]))
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


agent = DeepQLearningTensorflow(env_, [16, 16, 32, 32])
agent.fit()




















