import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utilities
import tensorflow as tf
keras = tf.keras


generator = utilities.generator
optimizer_gen = keras.optimizers.Adam(learning_rate=2/10_000, beta_1=1/2)
train_loss_gen = keras.metrics.Mean(name='train_loss')


discriminator = utilities.discriminator
optimizer_dis = keras.optimizers.Adam(learning_rate=2/10_000, beta_1=1/2)
train_loss_dis = keras.metrics.Mean(name='train_loss')


def run_generator(z, training=False):
    z_g = tf.reshape(z, [-1, 1, 1, utilities.CODE_LENGTH])
    return generator(z_g, training=training)


def run_discriminator(x, training=False):
    x_d = tf.image.resize(x, [64, 64])
    return discriminator(x_d, training=training)


@tf.function
def learn(x_i):

    z_n = tf.random.uniform(shape=[utilities.BATCH_SIZE, utilities.CODE_LENGTH], minval=-1., maxval=1.)

    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_dis:

        x_g = run_generator(z_n, training=True)
        z_d1 = run_discriminator(x_i, training=True)
        z_d2 = run_discriminator(x_g, training=True)

        loss_d1 = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(z_d1), z_d1)
        loss_d2 = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(z_d2), z_d2)
        loss_dis = loss_d1 + loss_d2
        loss_gen = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(z_d2), z_d2)

    grad_gen = tape_gen.gradient(loss_gen, generator.trainable_variables)
    grad_dis = tape_dis.gradient(loss_dis, discriminator.trainable_variables)
    optimizer_gen.apply_gradients(zip(grad_gen, generator.trainable_variables))
    optimizer_dis.apply_gradients(zip(grad_dis, discriminator.trainable_variables))

    train_loss_dis(loss_dis)
    train_loss_gen(loss_gen)


(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32)[0: utilities.N_SAMPLES] / 255.
width = X_train.shape[1]
height = X_train.shape[2]
X_train_g = tf.data.Dataset.from_tensor_slices(
    np.expand_dims(X_train, axis=3)
).shuffle(1000).batch(utilities.BATCH_SIZE)


def train():
    for e in range(1, utilities.N_EPOCHS+1):
        for x_i in X_train_g:
            learn(x_i)
        print('Epoch %d | Discriminator Loss: %.3f | Generator Loss: %.3f'
              % (e, train_loss_dis.result(), train_loss_gen.result()))
        train_loss_dis.reset_states()
        train_loss_gen.reset_states()


train()


#  SHOW SOME RESULTS
Z = np.random.uniform(low=-1., high=1., size=[50, utilities.CODE_LENGTH]).astype(np.float32)
Y_s = run_generator(Z, training=False)
Y_s = np.squeeze((Y_s + 1.) * 1/2 * 255.).astype(np.uint8)
sns.set()
fig, ax = plt.subplots(nrows=5, ncols=10, figsize=[22, 8])
for i in range(5):
    for j in range(10):
        ax[i, j].imshow(Y_s[(i * 10) + j], cmap='gray')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
plt.show()
plt.clf()































































































































































































































































































































