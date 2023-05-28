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


class TD3Tensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape_1,
            hidden_shape_2,
            actor_update_rate=2,
            alpha=1 / 1_000,
            beta=1/500,
            gamma=999/1_000,
            tau=1/100,
            noise_decay=99/100,
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
        self.noise = 3
        # self.buffer = ReplayBufferZeros(max_size=1_000, s_dim=self.n_s, a_dim=self.n_a)
        self.buffer = ReplayBuffer(max_size=1_000)
        self.actor = self._build_actor(hidden_shape_1)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.target_actor = self._build_actor(hidden_shape_1)
        self.critic_1, self.critic_2 = self._build_critic(hidden_shape_2), self._build_critic(hidden_shape_2)
        self.critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_1, self.target_critic_2 = self._build_critic(hidden_shape_2), self._build_critic(hidden_shape_2)
        self.critic_loss_function = tf.losses.MeanSquaredError()
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
        model = keras.models.Model([input_state, input_action], output_layer)
        return model

    #  if tau=1 then soft update is hard update
    def _target_actor_soft_update(self, tau=1.):
        assert 0 <= tau <= 1
        self.target_actor.set_weights(
            tau * np.array(self.actor.get_weights(), dtype=object)
            + (1-tau) * np.array(self.target_actor.get_weights(), dtype=object)
        )

    #  if tau=1 then soft update is hard update
    def _target_critic_soft_update(self, tau=1.):
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

    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        mean = self.actor(s)[0]
        action = np.random.normal(loc=mean, scale=self.noise)
        return np.clip(action, self.min_action, self.max_action)

    def _store_transition(self, s, a, r, s_, d):
        # self.buffer.push(s, a, r, s_, int(d))
        self.buffer.remember(s, a, r, s_, int(d))

    # @tf.function
    def _train_actor(self, states):
        with tf.GradientTape() as actor_tape:
            actions = self.actor(states)
            s_a_predictions = self.critic_1([states, actions])
            loss = - tf.reduce_sum(s_a_predictions)
        actor_grads = actor_tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    # @tf.function
    def _train_critics(self, states, actions, rewards, states_, dones):
        with tf.GradientTape() as tape_c_1, tf.GradientTape() as tape_c_2:
            actions_ = tf.clip_by_value(
                self.target_actor(states_), self.min_action, self.max_action
            )
            target_next_state_values = tf.math.minimum(
                self.target_critic_1([states_, actions_]),
                self.target_critic_2([states_, actions_])
            )
            y_pred = rewards + self.gamma * target_next_state_values * (1 - dones)
            critic_1_pred = self.critic_1([states, actions])
            critic_2_pred = self.critic_2([states, actions])
            critic_1_loss = self.critic_loss_function(y_pred, critic_1_pred)
            critic_2_loss = self.critic_loss_function(y_pred, critic_2_pred)
        critic_1_grad = tape_c_1.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grad = tape_c_2.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))

    def _train(self, update_actor):
        if self.buffer.get_buffer_size() < self.batch_size:
            return
        # states, actions, rewards, states_, dones = self.buffer.sample(batch_size=self.batch_size)
        states, actions, rewards, states_, dones = self.buffer.get_buffer(
            batch_size=self.batch_size, randomized=True, cleared=False
        )
        rewards = np.reshape(rewards, [-1, 1])
        dones = np.reshape(dones, [-1, 1])
        self._train_critics(states, actions, rewards, states_, dones)
        self._target_critic_soft_update(self.tau)
        if update_actor:
            self._train_actor(states)
            self._target_actor_soft_update(self.tau)

    def fit(self, n_episodes=2_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            for i in range(1, self.env._max_episode_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                self._store_transition(s, a, r, s_, d)
                score += r
                self._train(update_actor=i % self.actor_update_rate == 0)
                if d or t:
                    if score >= -130:  # hardcoded for 'Pendulum-v1'
                        counter += 1
                    else:
                        counter = 0
                    break
                s = s_
            if counter >= 7:
                print('Environment solved.')
                if save_model:
                    self.actor.save('../data/td3_nn_1/main_nn')
                    print('Model saved.')
                return self
            if self.noise > 1/100:
                self.noise = self.noise * self.noise_decay
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            print('Episode %d | noise: %.3f | score: %.3f | avg score: %.3f' % (ep, self.noise, score, avg_scores[-1]))
            if ep % 200 == 0 and graph:
                print_graph(scores, avg_scores, 'scores', 'avg scores', 'TD3 ep %d' % ep)
        return self

    def buffer_test(self):
        for _ in range(5):
            s = self.env.reset()[0]
            a = self._choose_action(s)
            s_, r, d, t, _ = self.env.step(a)
            self._store_transition(s, a, r, s_, d)
        return self.buffer.get_buffer(batch_size=5, randomized=True, cleared=False)


# td3 = TD3Tensorflow(env_, [16, 16, 32], [16, 32, 64])
# td3.fit()
# states, actions, rewards, states_, dones = td3.buffer_test()
# actions_ = tf.clip_by_value(td3.target_actor(states_), td3.min_action, td3.max_action)
# target_next_state_values = tf.math.minimum(td3.target_critic_1([states_, actions_]), td3.target_critic_2([states_, actions_]))


class SACTensorflow:

    def __init__(
            self,
            env: gym.wrappers.time_limit.TimeLimit,
            hidden_shape_1,
            hidden_shape_2,
            hidden_shape_3,
            alpha=1 / 1_000,
            beta=1/1_000,
            gamma=999/1_000,
            batch_size=2 ** 6
    ):
        self.env = env
        self.n_s, self.n_a = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.min_action, self.max_action = self.env.action_space.low[0], self.env.action_space.high[0]
        self.gamma = gamma
        self.buffer = ReplayBufferZeros(max_size=2_000, s_dim=self.n_s, a_dim=self.n_a)
        self.batch_size = batch_size
        self.temperature = 1/10
        self.ema = tf.train.ExponentialMovingAverage(decay=995/1_000)
        self.actor = self._build_actor(hidden_shape_1)
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))
        self.v, self.target_v = self._build_v(hidden_shape_2), self._build_v(hidden_shape_2)
        self.v.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.q_1 = self._build_q(hidden_shape_3)
        self.q_2 = keras.models.clone_model(self.q_1)
        self.q_1.compile(optimizer=keras.optimizers.Adam(learning_rate=beta))
        self.q_2.compile(optimizer=keras.optimizers.RMSprop(learning_rate=beta))
        self.q_and_v_loss = keras.losses.MeanSquaredError()

    def _build_actor(self, hidden_shape):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        mean = keras.layers.Dense(units=self.n_a, activation='linear')(layer)
        log_std = tf.clip_by_value(
            keras.layers.Dense(units=self.n_a, activation='linear')(layer), self.min_action, self.max_action
        )
        model = keras.models.Model(input_layer, [mean, log_std])
        return model

    def _build_v(self, hidden_shape):
        input_layer = keras.layers.Input(shape=self.n_s)
        layer = input_layer
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        model = keras.models.Model(input_layer, output_layer)
        return model

    def _build_q(self, hidden_shape):
        state_input = keras.layers.Input(shape=self.n_s)
        action_input = keras.layers.Input(shape=self.n_a)
        layer = keras.layers.concatenate([state_input, action_input], axis=-1)
        for h in hidden_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        model = keras.models.Model([state_input, action_input], output_layer)
        return model

    def _update_target_v(self):
        self.ema.apply(self.v.trainable_variables)
        for t_v_param, v_param in zip(self.target_v.trainable_variables, self.v.trainable_variables):
            t_v_param.assign(self.ema.average(v_param))

    def _choose_action(self, s):
        s = tf.convert_to_tensor([s])
        mean, log_std = self.actor(s)
        mean, log_std = mean[0], log_std[0]
        distribution = tfp.distributions.Normal(loc=mean, scale=np.exp(log_std))
        action = np.tanh(distribution.sample())
        return action * self.max_action

    def _select_action(self, s):
        s = tf.convert_to_tensor([s])
        mean, log_std = self.actor(s)
        mean = np.array(mean[0])
        log_std = np.array(log_std[0])
        std = np.exp(log_std)
        distribution = tfp.distributions.Normal(loc=mean, scale=std)
        a = np.tanh(distribution.sample())
        return a

    def _store_transition(self, s, a, r, s_, d):
        self.buffer.push(s, a, r, s_, int(d))

    def _train(self):

        if self.buffer.get_buffer_size() < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.buffer.sample(batch_size=self.batch_size)
        dones = np.reshape(dones, [-1, 1])

        with tf.GradientTape() as actor_tape:
            means, log_std = self.actor(states)
            stds = tf.exp(log_std)
            distribution = tfp.distributions.Normal(loc=means, scale=stds)
            samples = distribution.sample()
            # sampled_actions = tf.tanh(samples) * self.max_action
            sampled_actions = tf.tanh(samples)
            log_prob = distribution.log_prob(samples) - tf.math.log(1. - tf.pow(sampled_actions, 2) + 1e-16)
            log_prob = tf.reduce_mean(log_prob, axis=-1, keepdims=True)
            q_1_values = self.q_1([states, sampled_actions])
            q_2_values = self.q_2([states, sampled_actions])
            actor_loss = - tf.reduce_mean(
                q_1_values - (self.temperature * log_prob)
            )
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        q_values = tf.math.minimum(q_1_values, q_2_values)
        with tf.GradientTape() as v_tape:
            state_values = self.v(states)
            q_state_values = tf.stop_gradient(q_values - self.temperature * log_prob)
            v_loss = self.q_and_v_loss(q_state_values, state_values)
        v_grad = v_tape.gradient(v_loss, self.v.trainable_variables)
        self.v.optimizer.apply_gradients(zip(v_grad, self.v.trainable_variables))

        q_hat = rewards + self.gamma * tf.stop_gradient(self.target_v(states_)) * (1 - dones)
        with tf.GradientTape() as q_1_tape:
            q_1_preds = self.q_1([states, actions])
            q_1_loss = self.q_and_v_loss(q_hat, q_1_preds)
        q_1_grad = q_1_tape.gradient(q_1_loss, self.q_1.trainable_variables)
        self.q_1.optimizer.apply_gradients(zip(q_1_grad, self.q_1.trainable_variables))
        with tf.GradientTape() as q_2_tape:
            q_2_preds = self.q_2([states, actions])
            q_2_loss = self.q_and_v_loss(q_hat, q_2_preds)
        q_2_grad = q_2_tape.gradient(q_2_loss, self.q_2.trainable_variables)
        self.q_2.optimizer.apply_gradients(zip(q_2_grad, self.q_2.trainable_variables))

        self._update_target_v()

    def fit(self, n_episodes=2_000, graph=True, save_model=False):
        scores, avg_scores = [], []
        counter = 0
        for ep in range(1, n_episodes+1):
            score = 0
            s = self.env.reset()[0]
            for i in range(1, self.env._max_episode_steps):
                a = self._choose_action(s)
                s_, r, d, t, _ = self.env.step(a)
                self._store_transition(s, a, r, s_, d)
                score += r
                if i % 2 == 0:
                    self._train()
                if d or t:
                    if score >= -130:  # hardcoded for 'Pendulum-v1'
                        counter += 1
                    else:
                        counter = 0
                    break
                s = s_
            if counter >= 7:
                print('Environment solved.')
                if save_model:
                    pass
                return self
            scores.append(score)
            avg_scores.append(np.sum(scores[-50:]) / len(scores[-50:]))
            print('Episode %d | score: %.3f | avg score: %.3f' % (ep, score, avg_scores[-1]))
            if ep % 200 == 0 and graph:
                print_graph(scores, avg_scores, 'scores', 'avg scores', 'SAC ep %d' % ep)
        return self

    def buffer_test(self):
        for _ in range(5):
            s = self.env.reset()[0]
            a = self._choose_action(s)
            s_, r, d, t, _ = self.env.step(a)
            self._store_transition(s, a, r, s_, d)
        return self.buffer.sample(batch_size=5)


sac = SACTensorflow(env_, [16, 16, 32, 64], [16, 16, 32, 32], [16, 16, 16, 32])
# states, actions, rewards, states_, dones = sac.buffer_test()
# dones = np.reshape(dones, [-1, 1])
sac.fit()











