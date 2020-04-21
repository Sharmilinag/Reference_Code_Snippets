import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

BASE_MODEL = applications.VGG16(include_top=False, weights='imagenet')

# target dimensions of our images since the VGG16 model was trained on images with images of this dimension
IMG_WIDTH, IMG_HEIGHT = 224, 224

FINAL_DATA_PATH = 'data_split'
TRAINING_DATA_PATH = 'train'
VALIDATION_DATA_PATH = 'validation'
TEST_DATA_PATH = 'test'

MODEL_SAVE_PATH = 'clf_model.h5'

BATCH_SIZE = 16

EPOCHS = 2


def _get_bottleneck_features(model, dataset_type):
    """
    This function extracts the bottle neck features for the given dataset based on the given model
    """
    if dataset_type in ['train', 'validation']:
        #   creating the data generator with some augmented data in order to prevent over-fitting
        datagen = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest'
                                     )
        if dataset_type == 'train':
            data_dir = os.path.join(FINAL_DATA_PATH, TRAINING_DATA_PATH)
        else:
            data_dir = os.path.join(FINAL_DATA_PATH, VALIDATION_DATA_PATH)
    else:
        # we don't want to augment any data while evaluating the testing set
        datagen = ImageDataGenerator(rescale=1. / 255)
        data_dir = os.path.join(FINAL_DATA_PATH, TEST_DATA_PATH)

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

    nb_samples = len(generator.filenames)

    predict_size = int(math.ceil(nb_samples / BATCH_SIZE))

    bottleneck_features = model.predict_generator(generator, predict_size)

    return bottleneck_features


def _get_data_labels(model, generator, num_classes, dataset_type):
    """
    This function formats the data from the bottleneck features and converts the labels into categorical form for
    the given model, generator and num_classes
    """
    # get the training bottleneck features
    data = _get_bottleneck_features(model, dataset_type)

    # get the class lebels for the training data, in the original order
    labels = generator.classes

    # convert the training labels to categorical vectors
    labels = to_categorical(labels, num_classes=num_classes)
    return data, labels


def _evaluate_model(model, data, labels):
    """
    since function evaluates the model performance for the given data
    """
    (eval_loss, eval_accuracy) = model.evaluate(
        data, labels, batch_size=BATCH_SIZE, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))


def _plot_training_history(history):
    """
    This function plots the graph for the model train history
    """
    # summarize history for accuracy
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('history_accuracy.png')

    # summarize history for loss
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('history_loss.png')


def train_model():
    """
    This function trains a model based on the base model using transfer learning
    """
    # re-scale the images
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    # have the fully connected layer generator set up
    generator_top = datagen_top.flow_from_directory(os.path.join(FINAL_DATA_PATH, TRAINING_DATA_PATH),
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False
                                                    )

    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # get the training data and labels from the bottleneck features
    train_data, train_labels = _get_data_labels(BASE_MODEL, generator_top, num_classes, 'train')

    #   ## define the generator for the top validation dataset
    generator_top = datagen_top.flow_from_directory(os.path.join(FINAL_DATA_PATH, VALIDATION_DATA_PATH),
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False
                                                    )

    # get the validation data and labels from the bottleneck features
    validation_data, validation_labels = _get_data_labels(BASE_MODEL, generator_top, num_classes, 'validation')

    # set up the model with sigmoid as the final activation function and relu as the activation function for the flatten layer
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(MODEL_SAVE_PATH)

    _evaluate_model(model, validation_data, validation_labels)

    _plot_training_history(history)

    return model


def predict(image_path):
    """
    This function predicts the most probable class of the input image
    """
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # we would need to do this because the original images that were trained were trained within this range
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = BASE_MODEL.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(MODEL_SAVE_PATH)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    # get the class probabilities
    probabilities = model.predict_proba(bottleneck_prediction)

    label_id = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[label_id]

    # get the most probable prediction label
    print("Most Probable Breed: {0}".format(label))


def main():
    model = train_model()
    # predict('data_split/test/081.Greyhound/Greyhound_05538.jpg')


if __name__ == "__main__":
    main()
