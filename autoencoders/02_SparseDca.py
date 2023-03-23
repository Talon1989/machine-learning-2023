import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
keras = tf.keras
from Autoencoder import DAC
import param


np.random.seed(1000)
tf.random.set_seed(1000)


# CODE_LENGTH = 576
(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32)[0: param.N_SAMPLES] / 255.
width = X_train.shape[1]
height = X_train.shape[2]
#  selects block of shuffled data and returns batches at every call
X_train_g = tf.data.Dataset.from_tensor_slices(
    np.expand_dims(X_train, axis=3)
).shuffle(1000).batch(param.BATCH_SIZE)


model = DAC(param.CODE_LENGTH, sparse=True)
optimizer = keras.optimizers.Adam(1/1_000)
train_loss = keras.metrics.Mean(name='train_loss')


@tf.function
def learn(images):
    with tf.GradientTape() as tape:
        loss = keras.losses.MSE(model.r_images(images), model(images))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)


def train():
    for ep in range(1, param.N_EPOCHS+1):
        for x_i in X_train_g:
            learn(x_i)
        print('Epoch %d | Loss: %.3f' % (ep, train_loss.result()))
        train_loss.reset_states()


train()


codes = model.encoder(np.expand_dims(X_train, axis=3))
print('Code mean: %.3f\nCode std: %.3f' % (np.mean(codes), np.std(codes)))


Xs = np.reshape(X_train[0: param.BATCH_SIZE], [param.BATCH_SIZE, width, height, 1])
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





















































































































































































































































































