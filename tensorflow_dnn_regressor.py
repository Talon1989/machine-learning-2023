import numpy as np
import tensorflow as tf
keras = tf.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from CustomUtilities import print_graph


dataset = pd.read_csv('data/iris.csv')
X_ = dataset.iloc[:, 0:-2].to_numpy()
y_ = dataset.iloc[:, -2].to_numpy()
y_ = np.reshape(y_, [-1, 1])

X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=1/2, shuffle=True)


class Regressor:

    def __init__(self, n_input, h_shape, alpha=1/1_000, batch_size=32):
        self.n_input = n_input
        self.h_shape = h_shape
        self.batch_size = batch_size
        self.dnn = self._build_nn()
        self.dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=alpha))

    def _build_nn(self):
        input_layer = keras.layers.Input(shape=self.n_input)
        layer = input_layer
        for h in self.h_shape:
            layer = keras.layers.Dense(units=h, activation='relu')(layer)
        output_layer = keras.layers.Dense(units=1, activation='linear')(layer)
        model = keras.Model(input_layer, output_layer)
        return model

    def predict(self, x):
        if len(x.shape) == 1:
            return self.dnn(np.expand_dims(x, axis=0))
        return self.dnn(x)

    def train(self, X, y, n_epochs=3_000):
        losses, avg_losses = [], []
        for ep in range(1, n_epochs+1):
            batch = np.arange(len(y))
            loss_value = tf.constant(0., dtype=tf.float32)
            for idx in range(0, len(y), self.batch_size):
                minibatch = batch[idx: idx+self.batch_size]
                with tf.GradientTape() as tape:
                    y_pred = self.dnn(X[minibatch])
                    loss = tf.reduce_mean((y[minibatch] - y_pred) ** 2)
                    # loss = tf.losses.MSE(y[minibatch], y_pred)
                grads = tape.gradient(loss, self.dnn.trainable_variables)
                self.dnn.optimizer.apply_gradients(zip(grads, self.dnn.trainable_variables))
                loss_value += loss
                # loss_value += tf.reduce_sum(loss)
            losses.append(loss_value.numpy())
            avg_losses.append(np.sum(losses[-50:]) / len(losses[-50:]))
            if ep % 10 == 0:
                print('Epoch %d | loss: %.4f' % (ep, losses[-1]))
                # print('Epoch %d | loss: %s' % (ep, 'potato'))
            if ep % 1000 == 0:
                print_graph(losses, avg_losses, 'loss', 'avg loss', 'Tensorflow regressor on iris dataset')
        return self


regressor = Regressor(X_.shape[1], [2**4, 2**4, 2**5])
regressor.train(X_train, y_train, n_epochs=2_000)

print(
    r2_score(regressor.predict(X_test), y_test)
)













































































































































































































































































































