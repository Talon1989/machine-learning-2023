import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras
from sklearn.preprocessing import MinMaxScaler
from CustomUtilities import print_simple_graph
from CustomUtilities import print_graph


N_SAMPLES = 3175
SEQUENCE_LENGTH = 15


df = pd.read_csv('../data/sunspots.csv', header=None).dropna()
data = df[3].values[:N_SAMPLES - SEQUENCE_LENGTH].astype(np.float32)

#  since LSTM works with tanh it's helpful to normalize values
mmscaler = MinMaxScaler(feature_range=(-1., 1.))
data = mmscaler.fit_transform(data.reshape([-1, 1]))
# print_simple_graph(data)

#  prepare the dataset (2600 datapoints)
X_ts = np.zeros(shape=[N_SAMPLES - SEQUENCE_LENGTH, SEQUENCE_LENGTH, 1], dtype=np.float32)
Y_ts = np.zeros(shape=(N_SAMPLES - SEQUENCE_LENGTH, 1), dtype=np.float32)
for i in range(0, data.shape[0] - SEQUENCE_LENGTH):
    X_ts[i] = data[i: i + SEQUENCE_LENGTH]
    Y_ts[i] = data[i + SEQUENCE_LENGTH]
    X_ts_train = X_ts[0:2_600, :]
    Y_ts_train = Y_ts[0:2_600]
    X_ts_test = X_ts[2_600: N_SAMPLES, :]
    Y_ts_test = Y_ts[2_600: N_SAMPLES]

#  LSTM layer containing four cells
#  stateful=True forces TensorFlow/Keras not to reset the state after each batch (default is False)
model = keras.models.Sequential([
    keras.layers.LSTM(units=4, stateful=True, batch_input_shape=(20, SEQUENCE_LENGTH, 1)),
    keras.layers.Dense(units=1, activation='tanh')
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1/1_000, decay=1/10_000),
    loss=keras.losses.MSE,
    metrics=['mse']
)

model.fit(X_ts_train, Y_ts_train, batch_size=20, epochs=100, shuffle=False, validation_data=(X_ts_test, Y_ts_test))


# predicted_value = model.predict(X_ts_train)
# print_graph(predicted_value, Y_ts_test, 'predicted values', 'true values', 'LSTM sunspots', 'time steps', scatter=False)



































































































































































































