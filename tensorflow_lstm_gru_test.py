import numpy as np
import tensorflow as tf
keras = tf.keras
from keras.datasets import imdb


NUM_DISTINCT_WORDS = 5_000
EMBEDDING_OUTPUT_DIMS = 15
MAX_SEQUENCE_LENGTH = 300
BATCH_SIZE = 128
NUM_EPOCHS = 5


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_DISTINCT_WORDS)

#  pads all chars in each element in x_train and x_test to be 300 with value=0
padded_inputs_train = keras.preprocessing.sequence.pad_sequences(sequences=x_train, maxlen=MAX_SEQUENCE_LENGTH, value=0.)
padded_inputs_test = keras.preprocessing.sequence.pad_sequences(sequences=x_test, maxlen=MAX_SEQUENCE_LENGTH, value=0.)

model = keras.models.Sequential()
#  Word embedding is a technique used in natural language processing (NLP) that represents words as numerical vectors,
#  with the goal of capturing the meaning and relationships between words in a high-dimensional space.
model.add(
    keras.layers.Embedding(input_dim=NUM_DISTINCT_WORDS, output_dim=EMBEDDING_OUTPUT_DIMS, input_length=MAX_SEQUENCE_LENGTH)
)
# model.add(keras.layers.LSTM(units=10))
model.add(keras.layers.GRU(units=10))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
# model.summary()


history = model.fit(
    x=padded_inputs_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=True,
    validation_split=0.20
)
test_results = model.evaluate(padded_inputs_test, y_test, verbose=False)
print('Test results - Loss: %.4f - Accuracy: %.4f' % (test_results[0], test_results[1]))























































































































































































































































































































































































































