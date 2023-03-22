import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from CustomUtilities import onehot_transformation
import tensorflow as tf
keras = tf.keras


mnist_v2 = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist_v2.load_data()
y_train = onehot_transformation(y_train)
y_test = onehot_transformation(y_test)
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]


cnn = keras.Sequential([
    keras.layers.Conv2D(
        filters=32, kernel_size=[5, 5], strides=[1, 1],
        padding='same', data_format='channels_last', activation='relu'
    ),
    keras.layers.MaxPool2D([2, 2]),
    keras.layers.Conv2D(
        filters=64, kernel_size=[5, 5], strides=[1, 1],
        padding='same', activation='relu'
    ),
    keras.layers.MaxPool2D([2, 2]),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(units=10, activation='softmax')
])
cnn.build(input_shape=(None, 28, 28, 1))
cnn.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
cnn.fit(X_train, y_train, batch_size=10, epochs=2)


print('predicting first train data')
print(
    np.argmax(
        cnn.predict(X_train[0:1])[0]
    )
)