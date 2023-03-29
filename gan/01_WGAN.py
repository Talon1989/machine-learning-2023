import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import *
import tensorflow as tf
keras = tf.keras


# N_SAMPLES = 10240
# N_EPOCHS = 50
N_CRITIC = 5
# BATCH_SIZE = 64
# CODE_LENGTH = 256


generator = generator
optimizer_gen = keras.optimizers.Adam(learning_rate=5/10_000, beta_1=1/2, beta_2=9/10)
train_loss_gen = keras.metrics.Mean(name='train_loss')


discriminator = discriminator
optimizer_dis = keras.optimizers.Adam(learning_rate=5/10_000, beta_1=1/2, beta_2=9/10)
train_loss_dis = keras.metrics.Mean(name='train_loss')


def run_generator(z, training=False):
    z_g = tf.reshape(z, [-1, 1, 1, CODE_LENGTH])
    return generator(z_g, training=training)


def run_discriminator(x, training=False):
    x_d = tf.image.resize(x, [64, 64])
    return discriminator(x_d, training=training)


def run_model(x_i, z_n, training=False):
    x_g = run_generator(z_n, training=training)
    z_d1 = run_discriminator(x_i, training=training)
    z_d2 = run_discriminator(x_g, training=training)
    # print(z_d2.shape)
    # print(z_d1.shape)
    # print()
    loss_discriminator = tf.reduce_mean(z_d2 - z_d1)
    loss_generator = tf.reduce_mean(-z_d2)
    return loss_discriminator, loss_generator


# @tf.function
def learn_discriminator(x_i):
    # z_n = tf.random.uniform([BATCH_SIZE, CODE_LENGTH], -1., 1.)
    z_n = tf.random.uniform([x_i.shape[0], CODE_LENGTH], -1., 1.)
    with tf.GradientTape() as tape:
        loss_discriminator, _ = run_model(x_i, z_n, training=True)
    grad_discriminator = tape.gradient(loss_discriminator, discriminator.trainable_variables)
    optimizer_dis.apply_gradients(zip(grad_discriminator, discriminator.trainable_variables))
    for v in discriminator.trainable_variables:
        v.assign(tf.clip_by_value(v, -0.01, 0.01))
    train_loss_dis(loss_discriminator)


# @tf.function
def learn_generator():
    z_n = tf.random.uniform([BATCH_SIZE, CODE_LENGTH], -1., 1.)
    x_g = tf.zeros([BATCH_SIZE, width, height, 1])  # don't need discriminator values of true distribution
    with tf.GradientTape() as tape:
        _, loss_generator = run_model(x_g, z_n, training=True)
    grad_generator = tape.gradient(loss_generator, generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(grad_generator, generator.trainable_variables))
    train_loss_gen(loss_generator)


(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32)[0: N_SAMPLES] / 255.
width = X_train.shape[1]
height = X_train.shape[2]
X_train_g = tf.data.Dataset.from_tensor_slices(
    np.expand_dims(X_train, axis=3)
).shuffle(1000).batch(BATCH_SIZE)
# ).shuffle(1000).batch(N_CRITIC * BATCH_SIZE)


def train():
    for e in range(N_EPOCHS):
        for x_i in X_train_g:
            # for i in range(N_CRITIC):
            #     learn_discriminator(x_i[i * BATCH_SIZE: (i+1) * BATCH_SIZE])
            learn_discriminator(x_i)
            learn_generator()
        print('Epoch %d | Discriminator Loss: %.3f | Generator Loss: %.3f'
              % (e, train_loss_dis.result(), train_loss_gen.result()))
        train_loss_dis.reset_states()
        train_loss_gen.reset_states()


train()


#  SHOW SOME RESULTS
Z = np.random.uniform(low=-1., high=1., size=[50, CODE_LENGTH]).astype(np.float32)
Y_s = run_generator(Z, training=False)
Y_s = np.squeeze((Y_s + 1.) * 1/2 * 255.).astype(np.uint8)
sns.set()
fig, ax = plt.subplots(nrows=5, ncols=10, figsize=[22, 8])
for i in range(5):
    for j in range(10):
        ax[i, j].imshow(Y_s[i + j], cmap='gray')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.show()
plt.clf()
















































































































































































































































