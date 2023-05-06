import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
keras = tf.keras
from CustomUtilities import print_graph


"""
based on
https://www.tensorflow.org/tutorials/images/transfer_learning
dogs and cats classification from google MobileNet-V2 model
"""


URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
BATCH_SIZE = 2**5
IMG_SIZE = [160, 160]


#  DOWNLOAD LABELLED IMAGES AND CREATE TRAIN / VALIDATION DATASETS

path_to_zip = keras.utils.get_file('cats_and_dogs_zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dataset = keras.utils.image_dataset_from_directory(
    directory=os.path.join(PATH, 'train'), shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)
validation_dataset = keras.utils.image_dataset_from_directory(
    directory=os.path.join(PATH, 'validation'), shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)
class_names = train_dataset.class_names


def show_some_vanilla_images(n=9):
    assert np.sqrt(n) % 1 == 0
    plt.figure(figsize=[10, 10])
    for images, labels in train_dataset.take(1):
        for i in range(n):
            plt.subplot(int(np.sqrt(n)), int(np.sqrt(n)), i+1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')
    plt.show()
    plt.clf()


#  CREATE SOME TEST SET  (32 BATCHES, WE TAKE 6 FOR THE TEST)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)


#  AUTOTUNING TO USE BUFFERED PREFETCHING TO AVOID I/O BLOCKING

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


def show_some_image_augmentations(n=9):
    assert np.sqrt(n) % 1 == 0
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=[10, 10])
        image = image[0]
        for i in range(n):
            plt.subplot(int(np.sqrt(n)), int(np.sqrt(n)), i+1)
            augmented_image = data_augmentation(tf.expand_dims(image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')
    plt.show()
    plt.clf()


###################################################################################


#  GET THE BASE MODEL

mobile_net_v2 = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + [3, ],
    include_top=False,
    weights='imagenet'
)
# image_batch, label_batch = next(iter(train_dataset))
# feature_batch = mobile_net_v2(image_batch)
# print(feature_batch.shape)


#  FREEZE THE CONVOLUTIONAL BASE

#  When you set layer.trainable = False
#  the BatchNormalization layer will run in inference mode, and will not update its mean and variance statistics.

mobile_net_v2.trainable = False


#  SAMPLE DIVERSIFICATION

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip(mode='horizontal'),
    keras.layers.RandomRotation(factor=0.2)
])


#  ADD A CLASSIFICATION LAYERS FOR OUTPUT

global_average_layer = keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
prediction_layer = keras.layers.Dense(units=1, activation='linear')
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)


#  MODEL BUILDING

inputs = keras.Input(shape=mobile_net_v2.input_shape[1:])  # first element is None
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)  # turns range [-1, 1] into [0, 255]
x = mobile_net_v2(x, training=False)
x = global_average_layer(x)
x = keras.layers.Dropout(2/10)(x)
outputs = prediction_layer(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1/10_000),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)


#  TRAINING

epochs = 10
loss, accuracy = model.evaluate(validation_dataset)
print('Initial Loss : %.3f\nInitial Accuracy: %.3f' % (loss, accuracy))
history = model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs
)
print_graph(history.history['loss'], history.history['val_loss'], 'Training Accuracy', 'Validation Accuracy',
            'Transfer Learning Accuracy', scatter=False)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


########################################################################################


#  FINE-TUNING:
#  One way to increase performance even further is to train (or "fine-tune")
#  the weights of the top layers of the pre-trained model alongside the training of the classifier you added.
#  The training process will force the weights to be tuned from generic feature maps to features
#  associated specifically with the dataset

mobile_net_v2.trainable = True  # unlock training for all layers
print('Number of layer in the base model: %d' % len(mobile_net_v2.layers))
fine_tune_at = 100
for l in mobile_net_v2.layers[:fine_tune_at]:  # freeze all layers before 'fine_tune_at'
    l.trainable = False
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1/100_000),  # lower the learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs
history_fine = model.fit(
    train_dataset, validation_data=validation_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1]
)
# print_graph(history_fine.history['loss'], history_fine.history['val_loss'], 'Training Accuracy', 'Validation Accuracy',
#             'Transfer Learning Accuracy', scatter=False)
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()
plt.clf()


















































































































































































