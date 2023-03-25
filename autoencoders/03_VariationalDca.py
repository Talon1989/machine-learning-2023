import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
keras = tf.keras
import param


class VarDac(keras.Model):

    def __init__(self, width, height):
        super(VarDac, self).__init__()
        self.width = width
        self.height = height
        #  encoder layers
        self.e1 = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.e2 = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')
        self.e3 = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')
        self.flatten = keras.layers.Flatten()
        self.code_mean = keras.layers.Dense(units=self.width*self.height, activation='linear')
        self.code_log_variance = keras.layers.Dense(units=self.width*self.height, activation='linear')
        #  decoder layers
        self.d1 = keras.layers.Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.d2 = keras.layers.Conv2DTranspose(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.d3 = keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 3], activation='linear', padding='same')

    def r_images(self, x):
        return tf.image.resize(x, [32, 32])

    def encoder(self, x):
        e1 = self.e1(self.r_images(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        code_input = self.flatten(e3)
        mu = self.code_mean(code_input)
        std = tf.sqrt(tf.exp(self.code_log_variance(code_input)))
        normal_samples = tf.random.normal(
            mean=0., stddev=1., shape=[param.BATCH_SIZE, self.width*self.height]
        )
        z = (normal_samples * std) + mu
        return z, mu, std

    def decoder(self, z):
        d1 = self.d1(tf.reshape(z, [-1, 7, 7, 16]))
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        return d3, keras.activations.sigmoid(d3)

    def call(self, x):
        code, mu, std = self.encoder(x)
        logits, x_hat = self.decoder(code)
        return logits, mu, std, x_hat


(X_train, _), (_, _) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32)[0: param.N_SAMPLES] / 255.
width = X_train.shape[1]
height = X_train.shape[2]
#  selects block of shuffled data and returns batches at every call
X_train_g = tf.data.Dataset.from_tensor_slices(
    np.expand_dims(X_train, axis=3)
).shuffle(1000).batch(param.BATCH_SIZE)


model = VarDac(width, height)
optimizer = keras.optimizers.Adam(learning_rate=1/1_000)
train_loss = keras.metrics.Mean(name='train_loss')


@tf.function
def learn(images):
    with tf.GradientTape() as tape:
        logits, mu, std, _ = model(images)
        loss_r = tf.nn.sigmoid_cross_entropy_with_logits(labels=images, logits=logits)
        kl_divergence = 1/2 * tf.reduce_sum(
            tf.math.square(mu) + tf.math.square(std) - tf.math.log(tf.math.square(std) + 1e-8) - 1,
            axis=1
        )
        loss = tf.reduce_sum(loss_r) + kl_divergence
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)


def train():
    for e in range(1, param.N_EPOCHS+1):
        for x_i in X_train_g:
            learn(x_i)
        print('Epoch %d | Loss %.3f' % (e, train_loss.result()))
        train_loss.reset_states()


train()


Xs = np.reshape(X_train[0: param.BATCH_SIZE], [param.BATCH_SIZE, width, height, 1])
_, _, _, x_pred = model(Xs)
Ys = np.squeeze(x_pred * 255.)
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



























































































































































































