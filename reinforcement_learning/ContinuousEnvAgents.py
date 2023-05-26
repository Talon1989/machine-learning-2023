import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability

from Utilities import ReplayBuffer
from Utilities import ReplayBufferZeros
from CustomUtilities import print_graph
import tensorflow as tf
import tensorflow_probability as tfp

keras = tf.keras

np.random.seed(42)


env_ = gym.make('Pendulum-v1')


#  try to use Vanilla ReplayBuffer


class TD3Tensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape_1,
            hidden_shape_2,
            actor_update_rate=2,
            alpha=1 / 1_000,
            beta=1/500,
            gamma=99 / 100,
            tau=1/100,
            noise_decay=9_999 / 10_000,
            batch_size=2 ** 6
    ):
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.min_action, self.max_action = self.env.action_space.low[0], self.env.action_space.high[0]
        self.actor_update_rate = actor_update_rate
        self.gamma = gamma
        self.tau = tau
        self.noise_decay = noise_decay
        self.batch_size = batch_size
        self.buffer = ReplayBufferZeros(max_size=1_000, s_dim=self.n_s, a_dim=self.n_a)
        self.actor = self._build_actor(hidden_shape_1)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.target_actor = self._build_actor(hidden_shape_1)
        self.critic_1, self.critic_2 = self._build_critic(hidden_shape_2), self._build_critic(hidden_shape_2)
        self.critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_1, self.target_critic_2 = self._build_critic(hidden_shape_2), self._build_critic(hidden_shape_2)
        self._target_actor_soft_update()
        self._target_critic_soft_update()

    def _build_actor(self, hidden_shape):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=self.n_a, activation='tanh')(layer)
        output_layer = tf.math.multiply(output_layer, self.max_action)
        model = keras.models.Model(input_layer, output_layer)
        return model

    def _build_critic(self, hidden_shape):
        input_state = keras.layers.Input(shape=self.n_s)
        input_action = keras.layers.Input(shape=self.n_a)
        input_layer = keras.layers.concatenate([input_state, input_action], axis=-1)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    #  if tau=1 then soft update is hard update
    def _target_actor_soft_update(self, tau=1):
        assert 0 <= tau <= 1
        self.target_actor.set_weights(
            tau * np.array(self.actor.get_weights(), dtype=object)
            + (1-tau) * np.array(self.target_actor.get_weights(), dtype=object)
        )

    #  if tau=1 then soft update is hard update
    def _target_critic_soft_update(self, tau=1):
        assert 0 <= tau <= 1
        self.target_critic_1.set_weights(
            tau * np.array(self.critic_1.get_weights(), dtype=object)
            + (1 - tau) * np.array(self.target_critic_1.get_weights(), dtype=object)
        )
        self.target_critic_2.set_weights(
            tau * np.array(self.critic_2.get_weights(), dtype=object)
            + (1 - tau) * np.array(self.target_critic_2.get_weights(), dtype=object)
        )

    def load_actor(self, filepath='../data/td3_nn_1/main_nn'):
        self.actor = keras.models.load_model(filepath)

    def choose_action(self, s):
        pass


td3 = TD3Tensorflow(env_, [16, 16, 32], [16, 32, 64])





















