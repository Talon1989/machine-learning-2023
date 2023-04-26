import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
keras = tf.keras


"""
based on
https://www.tensorflow.org/tutorials/images/transfer_learning
dogs and cats classification from google MobileNet-V2 model
"""


URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
BATCH_SIZE = 2**5
IMG_SIZE = [160, 160]

path_to_zip = keras.utils.get_file('cats_and_dogs_zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_directory = os.path.join(PATH, 'train')
validation_directory = os.path.join(PATH, 'validation')
train_dataset = keras.utils.image_dataset_from_directory(
    directory=train_directory, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
)
validation_dataset = keras.utils.image_dataset_from_directory(
    directory=validation_directory, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
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


# show_some_vanilla_images(16)


#  create some test set  (32 batches, we take 6 for the test)
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)


#  autotuning to use buffered prefetching to avoid i/o blocking
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


#  sample diversification
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip(mode='horizontal'),
    keras.layers.RandomRotation(factor=0.2)
])


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


#  get the base model
mobile_net_v2 = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + [3, ],
    include_top=False,
    weights='imagenet'
)









































































































































































































