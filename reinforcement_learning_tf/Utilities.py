import numpy as np


# custom made for dql
class ReplayBuffer:

    def __init__(self, max_size=1_000):
        self.max_size = max_size
        self.states, self.actions, self.rewards, self.states_, self.dones = [], [], [], [], []

    def get_buffer_size(self):
        assert len(self.states) == len(self.actions) == len(self.rewards)
        return len(self.actions)

    def remember(self, s, a, r, s_, done):
        if len(self.states) > self.max_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.states_[0]
            del self.dones[0]
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.rewards, self.states_, self.dones = [], [], [], [], []

    def get_buffer(self, batch_size, randomized=True, cleared=False, return_bracket=False):
        assert batch_size <= self.max_size + 1
        indices = np.arange(self.get_buffer_size())
        if randomized:
            np.random.shuffle(indices)
        buffer_states = np.squeeze([self.states[i] for i in indices][0: batch_size])
        buffer_actions = [self.actions[i] for i in indices][0: batch_size]
        buffer_rewards = [self.rewards[i] for i in indices][0: batch_size]
        buffer_states_ = np.squeeze([self.states_[i] for i in indices][0: batch_size])
        buffer_dones = [self.dones[i] for i in indices][0: batch_size]
        if cleared:
            self.clear()
        if return_bracket:
            for i in range(batch_size):
                buffer_actions[i] = np.array(buffer_actions[i])
                buffer_rewards[i] = np.array([buffer_rewards[i]])
                buffer_dones[i] = np.array([buffer_dones[i]])
            return np.array(buffer_states), np.array(buffer_actions), np.array(buffer_rewards), np.array(buffer_states_), np.array(buffer_dones)
            # return tuple(np.array(buffer_states)), tuple(np.array(buffer_actions)), tuple(np.array(buffer_rewards)), tuple(np.array(buffer_states_)), tuple(np.array(buffer_dones))
        return np.array(buffer_states), np.array(buffer_actions), np.array(buffer_rewards), np.array(buffer_states_), np.array(buffer_dones)

    def get_buffer_pgm(self, discrete_actions=True):
        import tensorflow._api.v2.compat.v1 as tf
        tf.disable_v2_behavior()
        states = tf.convert_to_tensor(np.array(self.states, dtype=np.float32))
        if discrete_actions:
            actions = tf.one_hot(np.array(self.actions, dtype=np.int8), depth=np.unique(self.actions).shape[0])
        else:
            actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        self.clear()
        return states, actions, rewards

    def __len__(self):
        return len(self.actions)


class PrioritizedExperienceReplayBuffer:

    def __init__(self, max_size=1_000):
        self.max_size = max_size
        self.states, self.actions, self.rewards, self.states_, self.dones = [], [], [], [], []

    def get_buffer_size(self):
        assert len(self.states) == len(self.actions) == len(self.rewards)
        return len(self.actions)

    def remember(self, s, a, r, s_, done):
        if len(self.states) > self.max_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.states_[0]
            del self.dones[0]
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_.append(s_)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.rewards, self.states_, self.dones = [], [], [], [], []

    def get_buffer_old(self, batch_size, main_nn, target_nn, gamma, alpha, beta, bias_correction=False, cleared=True):
        import tensorflow as tf
        assert batch_size <= self.max_size
        assert 0 <= alpha <= 1
        # indices = np.arange(self.get_buffer_size())
        states = np.squeeze(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        states_ = np.squeeze(self.states)
        dones = np.array(self.dones)
        y_pred = main_nn(states)
        y_pred_a = y_pred.numpy()[np.arange(y_pred.shape[0]), actions]
        # y_a = rewards + gamma * np.max(target_nn(states_), axis=1) * (1 - dones)
        y_a = rewards + gamma * np.max(target_nn(states_), axis=1)
        delta = np.abs(y_a - y_pred_a) + 1e-8
        denominator = np.sum(delta ** alpha)
        probas = (delta ** alpha) / denominator
        if not bias_correction:
            indices = np.random.choice(self.get_buffer_size(), batch_size, p=probas, replace=False)
            states_buffer = states[indices]
            actions_buffer = actions[indices]
            rewards_buffer = rewards[indices]
            states__buffer = states_[indices]
            dones_buffer = dones[indices]
        else:
            beta_start = 4/10
            importance_weights = (1 / (self.get_buffer_size() * probas)) ** np.max(beta_start, beta)
            indices = np.argsort(importance_weights)[::-1][0:batch_size]
            states_buffer = states[indices]
            actions_buffer = actions[indices]
            rewards_buffer = rewards[indices]
            states__buffer = states_[indices]
            dones_buffer = dones[indices]
        if cleared:
            self.clear()
        return states_buffer, actions_buffer, rewards_buffer, states__buffer, dones_buffer
        # y = np.copy(y_pred)
        # y[np.arange(y_pred.shape[0]), actions] = y_a
        # y = tf.convert_to_tensor(y, dtype=tf.float32)

    def get_buffer(self, batch_size, main_nn, target_nn, gamma, alpha, beta, bias_correction=False, cleared=True):
        import tensorflow as tf
        assert batch_size <= self.max_size
        assert 0 <= alpha <= 1
        # indices = np.arange(self.get_buffer_size())
        states = np.squeeze(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        states_ = np.squeeze(self.states)
        dones = np.array(self.dones)
        y_pred = main_nn(states)
        y_pred_a = y_pred.numpy()[np.arange(y_pred.shape[0]), actions]
        # y_a = rewards + gamma * np.max(target_nn(states_), axis=1) * (1 - dones)
        y_a = rewards + gamma * np.max(target_nn(states_), axis=1)
        delta = np.abs(y_a - y_pred_a) + 1e-8
        denominator = np.sum(delta ** alpha)
        probas = (delta ** alpha) / denominator
        indices = np.random.choice(self.get_buffer_size(), batch_size, p=probas, replace=False)
        states_buffer = states[indices]
        actions_buffer = actions[indices]
        rewards_buffer = rewards[indices]
        states__buffer = states_[indices]
        dones_buffer = dones[indices]
        if cleared:
            self.clear()
        if bias_correction:
            # beta_start = 4/10
            # importance_weights = (1 / (self.get_buffer_size() * probas[indices])) ** np.max([beta_start, beta])
            importance_weights = (1 / (self.get_buffer_size() * probas[indices])) ** beta
            return states_buffer, actions_buffer, rewards_buffer, states__buffer, dones_buffer, importance_weights
        return states_buffer, actions_buffer, rewards_buffer, states__buffer, dones_buffer
        # y = np.copy(y_pred)
        # y[np.arange(y_pred.shape[0]), actions] = y_a
        # y = tf.convert_to_tensor(y, dtype=tf.float32)


class ReplayBufferZeros:

    def __init__(self, max_size, s_dim, a_dim):
        self.states = np.zeros([max_size, s_dim], dtype=np.float32)
        self.actions = np.zeros([max_size, a_dim], dtype=np.float32)
        self.rewards = np.zeros([max_size], dtype=np.float32)
        self.states_ = np.zeros([max_size, s_dim], dtype=np.float32)
        self.dones = np.zeros([max_size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def push(self, s, a, r, s_, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.states_[self.ptr] = s_
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_buffer_size(self):
        return self.size

    def sample(self, batch_size=32, rewards_reshaped=True):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # temp_dict = dict(s=self.states[idxs],
        #                  s2=self.states_[idxs],
        #                  a=self.actions[idxs],
        #                  r=self.rewards[idxs],
        #                  d=self.dones[idxs])
        # print(temp_dict['r'])
        # print()
        # return temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1), temp_dict?['s2'], temp_dict['d']
        if rewards_reshaped:
            return self.states[idxs], self.actions[idxs], self.rewards[idxs].reshape([-1, 1]), self.states_[idxs], self.dones[idxs]
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.states_[idxs], self.dones[idxs]

    def clear(self):
        self.__init__(self.max_size, self.states.shape[1], self.actions.shape[1])


# https://github.com/samuelmat19/DDPG-tf2/blob/7852803d9e2a5ee4ca111d65056dcbdb7182b527/src/utils.py
class OUActionNoise:
    """
    Noise as defined in the DDPG algorithm
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):  # pylint: disable=too-many-arguments
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = None
        self.reset()

    def __call__(self):
        """
        The Ornstein-Uhlenbeck process is a mean-reverting stochastic process,
        which means that the values tend to stay around the mean value.
        Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        Return:
            The noise is being returned.
        """
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        """
        Reset the x_prev variable which is intrinsic to the algorithm
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Encoding:

    @staticmethod
    def onehot_encoding(x):
        onehot = np.zeros([len(np.unique(x)), len(x)])
        for idx, val in enumerate(x):
            onehot[int(val), idx] = 1
        return onehot.T
