import numpy as np
import tensorflow as tf
keras = tf.keras


"""
based on
https://github.com/PacktPublishing/Mastering-Machine-Learning-Algorithms-Second-Edition/blob/master/Chapter12/dca.py
"""


class DAC(keras.Model):

    def __init__(self, code_length, sparse=False):

        super(DAC, self).__init__()
        self.code_length = code_length

        #  encoder layers
        self.e1 = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], activation='relu', padding='same')
        self.e2 = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')
        self.e3 = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')
        self.flatten = keras.layers.Flatten()
        if not sparse:
            self.dense = keras.layers.Dense(units=self.code_length, activation='sigmoid')
        else:
            self.dense = keras.layers.Dense(
                units=self.code_length, activation='sigmoid', activity_regularizer=keras.regularizers.l1(1/10)
            )

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
        pixel_length = np.int32(np.sqrt(self.code_length))
        decoder_input = tf.reshape(z, [-1, pixel_length, pixel_length, 1])
        # decoder_input = tf.reshape(z, [-1, 16, 16, 1])
        d1 = self.d1(decoder_input)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        return self.d4(d3)

    def call(self, x):
        code = self.encoder(x)
        xhat = self.decoder(code)
        return xhat
