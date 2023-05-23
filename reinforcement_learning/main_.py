import numpy as np
import gym


env = gym.make('CartPole-v1')


def random_steps(environment: gym.wrappers.time_limit.TimeLimit, n_episodes=5):
    environment = gym.make(environment.unwrapped.spec.id, render_mode='human')
    for ep in range(1, n_episodes+1):
        environment.reset()
        while True:
            environment.render()
            action = environment.action_space.sample()
            _, _, d, t, _ = environment.step(action)
            if d or t:
                break
    environment.close()


#  testing tf.reduce_mean and keras.losses.MeanSquaredError functions


import tensorflow as tf
keras = tf.keras


mse = keras.losses.MeanSquaredError()


# y_true = tf.constant([1.0, 2.0, 3.0])
# y_pred = tf.constant([0.5, 1.5, 2.5])

y_true = tf.constant([
    [1.0, 2.0, 3.0],
    [1.1, 3.5, -1.2]
])
y_pred = tf.constant([
    [0.5, 1.5, 2.5],
    [1.0, 2.9, 0]
])


print(tf.reduce_mean((y_true - y_pred) ** 2))
print(mse(y_true, y_pred))




