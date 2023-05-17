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


examples, _ = tfds.load(
    name='ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True
)
train_examples, val_examples = examples['train'], examples['validation']


def print_examples(batch=3):
    for pt_examples, en_examples in train_examples.batch(batch).take(1):
        print('Examples in Portuguese:')
        print(pt_examples)
        # for pt in pt_examples.numpy():
        #     print(pt.decode('utf-8'))
        print()
        print('Examples in English:')
        print(en_examples)
        # for en in en_examples.numpy():
        #     print(en.decode('utf-8'))
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


######################################################################################


#  BUILD TOKENIZED BATCHES


#  The inputs are pairs of tokenized Portuguese and English sequences, (pt, en)
#  and the labels are the same English sequences shifted by 1,
#  this shift is so that at each location input en sequence, the label in the next token.
#  This is so that at each timestep the true value (predicted or not by the output) is input of next timestep,
#  now the model doesn't need to run sequentially, and we can use parallelism.
#  This is called 'Teacher Forcing'


#  builds tf.data.Dataset object set up to work with Keras, hardcoded to work with pt_en
def make_tokenized_batches(ds):

    def tokenize_batch(pt, en, max_tokens=128):
        pt = tokenizers.pt.tokenize(pt)
        pt = pt[:, :max_tokens]
        pt = pt.to_tensor()

        en = tokenizers.en.tokenize(en)
        en = en[:, :max_tokens + 1]
        en_inputs = en[:, :-1].to_tensor()  # drop [END] tokens
        en_labels = en[:, 1:].to_tensor()  # drop [START] tokens

        return (pt, en_inputs), en_labels

    return ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(tokenize_batch, tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)


train_batches = make_tokenized_batches(train_examples)
val_batches = make_tokenized_batches(val_examples)

######################################################################################


#  POSITIONAL EMBEDDING


def create_positional_encoding(length, depth):
    depth = depth / 2
    positions = np.reshape(np.arange(length), newshape=[-1, 1])
    depths = np.expand_dims(np.arange(depth), axis=0) / depth
    angle_rads = positions * (1 / (10_000 ** depths))
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def graph_pos_encoding():
    positional_encoding = create_positional_encoding(length=2**11, depth=2**9)
    print(positional_encoding)
    print(positional_encoding.shape)
    plt.pcolormesh(positional_encoding.numpy().T, cmap='RdBu')
    plt.xlabel('Position')
    plt.ylabel('Depth')
    plt.colorbar()
    plt.show()
    plt.clf()


class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, vocab_size, depth_model):
        super().__init__()
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=depth_model, mask_zero=True)
        self.positional_encoding = create_positional_encoding(length=2**11, depth=depth_model)
        self.depth_model = depth_model

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.depth_model, tf.float32))  # sets relative scale of embedding and pos encoding
        x = x + self.positional_encoding[tf.newaxis, :length, :]
        return x


pt_embedding = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), depth_model=2**9)
en_embedding = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), depth_model=2**9)


def print_embedded():
    for (pt, en), en_Labels in train_batches.take(1):
        en_emb = en_embedding(en)
        print(en_emb)


######################################################################################


#  MULTI HEAD ATTENTION LAYERS


class BaseAttention(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.add = keras.layers.Add()  # need to use Add() layer because '+' will not propagate
        self.layer_normalization = keras.layers.LayerNormalization()


class GlobalSelfAttention(BaseAttention):  # ENCODER MHA

    def call(self, x):
        attention_output = self.mha(
            query=x, key=x, value=x
        )
        x = self.add([x, attention_output])
        x = self.layer_normalization(x)
        return x


class CausalSelfAttention(BaseAttention):  # DECODER MMHA

    def call(self, x):
        attention_output = self.mha(
            query=x, key=x, value=x, use_causal_mask=True
        )
        x = self.add([x, attention_output])
        x = self.layer_normalization(x)
        return x


class CrossAttention(BaseAttention):  # DECODER MHA

    def call(self, x, context):
        attention_output, attention_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )
        # self.last_attention_scores = attention_scores
        x = self.add([x, attention_output])
        x = self.layer_normalization(x)
        return x


def print_attention_shape():

    for (pt, en), en_Labels in train_batches.take(1):

        pt_emb = pt_embedding(pt)
        print(pt_emb.shape)
        en_emb = en_embedding(en)
        print(en_emb.shape)

        # sample_ca = CrossAttention(num_heads=2, key_dim=512)
        # print(sample_ca(en_emb, pt_emb).shape)

        # sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
        # print(sample_gsa(pt_emb).shape)

        sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
        print(sample_csa(en_emb).shape)


######################################################################################


#  FEED FORWARD NETWORKS










































































































































































































































































































































































































































































































































