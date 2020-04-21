from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import models
from keras import layers
from keras import optimizers
import os
import matplotlib
import math
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMAGE_SIZE = 224
FINAL_DATA_PATH = 'data_split'
TRAINING_DATA_PATH = 'train'
VALIDATION_DATA_PATH = 'validation'
TEST_DATA_PATH = 'test'
#
TRAIN_BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 16


def _get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(FINAL_DATA_PATH, TRAINING_DATA_PATH),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(FINAL_DATA_PATH, VALIDATION_DATA_PATH),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=VALIDATION_BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

    return train_generator, validation_generator


def _plot_training_performance(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('training_validation_acc_loss.png')


def _validate_model_performance(model, generator):
    (eval_loss, eval_accuracy) = model.evaluate_generator(generator,
                                                          steps=generator.samples / generator.batch_size,
                                                          verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))


def train_model():
    # Load the VGG model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Freeze the layers except the last 4 layers
    for layer in base_model.layers[:-1]:          #[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in base_model.layers:
        print(layer, layer.trainable)

    train_generator, validation_generator = _get_data_generators()

    #   ##  total classes available
    num_classes = len(train_generator.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', train_generator.class_indices)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(base_model)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    # Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1)

    # Save the model
    model.save('small_last4.h5')

    #   ##  Plot Training and validation accuracy and loss
    _plot_training_performance(history)

    # #   ## Obtain Train and Validation Bottleneck Features
    # nb_train_samples = len(train_generator.filenames)
    # predict_size_train = int(math.ceil(nb_train_samples / TRAIN_BATCH_SIZE))
    # bottleneck_features_train = base_model.predict_generator(train_generator, predict_size_train)
    # nb_validation_samples = len(validation_generator.filenames)
    # predict_size_validation = int(math.ceil(nb_validation_samples / VALIDATION_BATCH_SIZE))
    # bottleneck_features_validation = model.predict_generator(validation_generator, predict_size_validation)

    #   ##  Check Model performance on Validation data
    _validate_model_performance(model, validation_generator)


train_model()
