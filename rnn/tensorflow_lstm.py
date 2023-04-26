import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from CustomUtilities import create_sequential_dataset
from CustomUtilities import print_graph
import tensorflow as tf
keras = tf.keras


tf.random.set_seed(42)
dataframe = pd.read_csv('../data/airline_passengers.cvs', usecols=[1], engine='python').dropna()
dataset = dataframe.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# train_size = int(len(dataset) * 0.67)
X_train, X_test = train_test_split(dataset, train_size=1/2, shuffle=False)
sequence_length = 3
X_train_seq, Y_train_next = create_sequential_dataset(dataset=X_train, look_back=sequence_length)
X_test_seq, Y_test_next = create_sequential_dataset(dataset=X_test, look_back=sequence_length)

#  reshape input to [samples, time_steps, features]
X_train_seq = np.reshape(X_train_seq, [X_train_seq.shape[0], 1, X_train_seq.shape[1]])
X_test_seq = np.reshape(X_test_seq, [X_test_seq.shape[0], 1, X_test_seq.shape[1]])

model = keras.models.Sequential([
    # keras.layers.LSTM(units=4, input_shape=(1, sequence_length)),
    keras.layers.GRU(units=4, input_shape=(1, sequence_length)),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
])
model.compile(
    loss=keras.losses.MSE,
    optimizer=keras.optimizers.Adam(learning_rate=5/10_000)
)
model.fit(X_train_seq, Y_train_next, epochs=100, batch_size=1, verbose=2)
predictions = model.predict(X_test_seq)
print_graph(predictions, Y_test_next, 'predicted data', 'true data', 'LSTM', 'month', False)
