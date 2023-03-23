import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
keras = tf.keras
from Autoencoder import DAC


N_SAMPLES = 1_000
# N_EPOCHS = 400
N_EPOCHS = 40
BATCH_SIZE = 200
CODE_LENGTH = 2**8


(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32)[0: N_SAMPLES] / 255.
width = X_train.shape[1]
height = X_train.shape[2]
#  selects block of shuffled data and returns batches at every call
X_train_g = tf.data.Dataset.from_tensor_slices(
    np.expand_dims(X_train, axis=3)
).shuffle(1000).batch(BATCH_SIZE)


model = DAC(CODE_LENGTH)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
train_loss = keras.metrics.Mean(name='train_loss')


def visualize_noise_data():
    Xs = np.reshape(X_train[0: 10], [10, width, height, 1])
    sns.set()
    fig, ax = plt.subplots(nrows=2, ncols=10, figsize=[18, 4])
    X_noise_gauss = np.clip(Xs + np.random.normal(0., 0.2, size=[10, width, height, 1]), a_min=0., a_max=1.)
    X_noise_dropout = Xs * np.random.binomial(n=1, p=0.5, size=[10, width, height, 1])
    for i in range(10):
        ax[0, i].imshow(np.squeeze(X_noise_gauss[i]), cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].imshow(X_noise_dropout[i], cmap='gray')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
    plt.show()
    plt.clf()


@tf.function
def learn(noisy_images, images):
    with tf.GradientTape() as tape:
        reconstruction = model(noisy_images)
        loss = keras.losses.MSE(model.r_images(images), reconstruction)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)


def train_noisy_gaussian():
    for ep in range(1, N_EPOCHS+1):
        for x_i in X_train_g:
            x_noisy = np.clip(x_i + np.random.normal(0., 0.2, size=[BATCH_SIZE, width, height, 1]), a_min=0., a_max=1.)
            learn(x_noisy, x_i)
        print('Epoch %d | Loss: %.3f' % (ep, train_loss.result()))
        train_loss.reset_states()


def train_noisy_dropout(p=1/5):
    for ep in range(1, N_EPOCHS+1):
        for x_i in X_train_g:
            dropout_noise = np.random.binomial(n=1, p=p, size=[BATCH_SIZE, width, height, 1])
            x_noisy = x_i * dropout_noise
            learn(x_noisy, x_i)
        print('Epoch %d | Loss: %.3f' % (ep, train_loss.result()))
        train_loss.reset_states()


train_noisy_gaussian()
# train_noisy_dropout()


codes = model.encoder(np.expand_dims(X_train, axis=3))
print('Code mean: %.3f\nCode std: %.3f' % (np.mean(codes), np.std(codes)))


Xs = np.reshape(X_train[0: BATCH_SIZE], [BATCH_SIZE, width, height, 1])
Ys = np.squeeze(model(Xs) * 255.)
sns.set()
fig, ax = plt.subplots(nrows=2, ncols=10, figsize=[18, 4])
for i in range(10):
    ax[0, i].imshow(np.squeeze(Xs[i]), cmap='gray')
    ax[0, i].set_xticks([])
    ax[0, i].set_yticks([])
    ax[1, i].imshow(Ys[i], cmap='gray')
    ax[1, i].set_xticks([])
    ax[1, i].set_yticks([])
plt.show()
plt.clf()














