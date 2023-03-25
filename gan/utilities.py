import numpy as np
import tensorflow as tf
keras = tf.keras


# N_SAMPLES = 5_000
N_SAMPLES = 2_000
CODE_LENGTH = 100
# N_EPOCHS = 50
N_EPOCHS = 20
BATCH_SIZE = 2**7


# input_layer = keras.Input(shape=[1, 1, CODE_LENGTH])
generator = keras.models.Sequential([
    keras.layers.Conv2DTranspose(
        input_shape=[1, 1, CODE_LENGTH], filters=2**10, kernel_size=[4, 4], padding='valid', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=3/10),
    keras.layers.Conv2DTranspose(
        filters=2**9, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2DTranspose(
        filters=2**8, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2DTranspose(
        filters=2**7, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2DTranspose(
        filters=1, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='tanh'
    )
])


discriminator = keras.models.Sequential([
    keras.layers.Conv2D(
        input_shape=[64, 64, 1], filters=2**7, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    # keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2D(
        filters=2**8, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2D(
        filters=2**9, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2D(
        filters=2**10, kernel_size=[4, 4], strides=[2, 2], padding='same', activation='linear'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(),
    keras.layers.Conv2D(
        filters=1, kernel_size=[4, 4], padding='valid', activation='linear'
    )
])
