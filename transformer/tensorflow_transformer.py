import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
keras = tf.keras


"""
based on:
https://www.tensorflow.org/text/tutorials/transformer
"""


#  DATA PREPPING


BUFFER_SIZE = 20_000
BATCH_SIZE = 2**6


examples, metadata = tfds.load(
    name='ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True
)
train_examples, val_examples = examples['train'], examples['validation']


def print_examples(batch=3):
    for pt_examples, en_examples in train_examples.batch(batch).take(1):
        print('Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()
        print('Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))
        print()


#  TOKENIZING THE SET

tokenizer_model = 'ted_hrlr_translate_pt_en_converter'
keras.utils.get_file(
    fname=f'{tokenizer_model}.zip',
    origin='https://storage.googleapis.com/download.tensorflow.org/models/%s.zip' % tokenizer_model,
    cache_dir='.',
    cache_subdir='',
    extract=True
)
tokenizers = tf.saved_model.load(tokenizer_model)


#  The tokenize method converts a batch of strings to a padded-batch of token IDs
def print_tokenize(batch=3):
    for _, en_examples in train_examples.batch(batch).take(1):
        print('Batch of strings:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))
        print('Batch of token IDs:')
        encoded = tokenizers.en.tokenize(en_examples)
        for row in encoded.to_list():
            print(row)
        print('Detokenized data:')
        decoder = tokenizers.en.detokenize(encoded)
        for line in decoder.numpy():
            print(line.decode('utf-8'))
        print('Text split into tokens:')
        tokens = tokenizers.en.lookup(encoded)
        print(tokens)
    print()


def graph_token_data():
    lengths = []
    for pt_examples, en_examples in train_examples.batch(2**10):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(pt_tokens.row_lengths())
        lengths.append(en_tokens.row_lengths())
    all_lengths = np.concatenate(lengths)
    plt.hist(all_lengths, np.linspace(0, 500, 101))
    plt.ylim(plt.ylim())
    max_length = max(all_lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title('Maximum tokens per example: %d' % max_length)
    plt.show()
    plt.clf()


def make_batches(ds):

    def prepare_batch(pt, en, max_tokens=128):
        pt = tokenizers.pt.tokenize(pt)
        pt = pt[:, :max_tokens]
        pt = pt.to_tensor()

        en = tokenizers.en.tokenize(en)
        en = en[:, :max_tokens + 1]
        en_inputs = en[:, :-1].to_tensor()  # drop [END] tokens
        en_labels = en[:, 1:].to_tensor()  # drop [START] tokens

        return (pt, en_inputs), en_labels

    return ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(prepare_batch, tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)


#  TEST DATASET

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)



























































































































































































































































































































































































































































































































































