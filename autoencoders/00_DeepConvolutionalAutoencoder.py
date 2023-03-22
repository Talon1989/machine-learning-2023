import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
keras = tf.keras


"""
based on
https://github.com/PacktPublishing/Mastering-Machine-Learning-Algorithms-Second-Edition/blob/master/Chapter12/dca.py
"""


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


class DAC(keras.Model):

    def __init__(self):

        super(DAC, self).__init__()

        #  encoder layers
        self.e1 = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.e2 = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')
        self.e3 = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(units=CODE_LENGTH, activation='sigmoid')

        #  decoder layers
        self.d1 = keras.layers.Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.d2 = keras.layers.Conv2DTranspose(filters=64, kernel_size=[3, 3], activation='relu', padding='same')
        self.d3 = keras.layers.Conv2DTranspose(filters=32, kernel_size=[3, 3], activation='relu', padding='same')
        self.d4 = keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 3], activation='sigmoid', padding='same')

    def r_images(self, x):
        return tf.image.resize(x, [32, 32])

    def encoder(self, x):
        e1 = self.e1(self.r_images(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        code_input = self.flatten(e3)
        return self.dense(code_input)

    def decoder(self, z):
        decoder_input = tf.reshape(z, [-1, 16, 16, 1])
        d1 = self.d1(decoder_input)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        return self.d4(d3)

    def call(self, x):
        code = self.encoder(x)
        xhat = self.decoder(code)
        return xhat


model = DAC()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
train_loss = keras.metrics.Mean(name='train_loss')


@tf.function
def learn(images):
    with tf.GradientTape() as tape:
        reconstruction = model(images)
        loss = keras.losses.MSE(model.r_images(images), reconstruction)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)


def train():
    for ep in range(1, N_EPOCHS+1):
        for x_i in X_train_g:
            learn(x_i)
        print('Epoch %d | Loss: %.3f' % (ep, train_loss.result()))
        train_loss.reset_states()


train()


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


















































































































































































































































































































































































































































































































