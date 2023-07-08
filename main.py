# Script with a classification of ASL signs using CNN
# Made for recrutation process for GoOnline Company
# author: Marianna Jucewicz
import pathlib
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(data_dir="data", batch_size=16, img_height=300, img_width=300):
    """
    The load_data function loads the data from a directory and returns three datasets:
        - train_ds: A dataset containing training images.
        - val_ds: A dataset containing validation images.
        - test_ds: A dataset containing testing images.

    :param data_dir: directory where the data is stored
    :param batch_size: Specify the number of images that are loaded into memory at once
    :param img_height: height of the image in pixels
    :param img_width: width of the image in pixels
    :return: A tuple of three batched tensorflow Datasets
    """

    train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_batches = tf.data.experimental.cardinality(val)
    test = val.take(val_batches // 2)
    val = val.skip(val_batches // 2)
    return train, val, test


def load_labels(data_dir="data"):
    """
    The load_labels function takes a data directory and returns the labels of the images in that directory.
    The function also returns the number of classes (labels) in that directory.

    :param data_dir: directory where the data is stored
    :return: A list of labels and the number of classes
    """
    p = pathlib.Path(data_dir)
    labels_list = []
    dirs = p.iterdir()

    for x in dirs:
        labels_list.append(x.parts[-1])

    classes = len(labels_list)
    labels_list.sort()
    return labels_list, classes


def prepare_model(num_classes, img_height=300, img_width=300):
    """
    The prepare_model function takes in the number of classes and returns a model.
    The model is a Sequential Keras Model with preprocessing layers.

    :param num_classes: Number of output classes
    :param img_height: Height of the input images
    :param img_width: Width of the image
    :return: A new model
    """

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
        ]
    )

    data_normalization = layers.Rescaling(1. / 255)

    new_model = tf.keras.Sequential([
        data_augmentation,

        data_normalization,

        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])
    return new_model


def train_model(model, train_ds, val_ds, epochs):
    """
    The train_model function trains the model on the training dataset and validates it on the validation dataset.

    :param model: model to be trained
    :param train_ds: training dataset
    :param val_ds: validation dataset
    :param epochs:  number of epochs to train for
    :return: history of training
    """

    training_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return training_history


def plot_learning_curves(training_history):
    """
    The plot_learning_curves function takes a history object and plots the training and validation accuracy/loss curves.

    :param training_history: history object with history of training a model
    """

    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()


def evaluate_model(model, test_ds):
    """
    The evaluate_model function takes a model and a test dataset as input.
    It then iterates over the test dataset, making predictions for each batch of data.
    The predictions are stored in an array called y_pred, while the true labels are stored in an array called y_true.
    Once all batches have been processed, we concatenate both arrays into single tensors.
    The results are used to calculate metrics: accuracy, precision, recall and f1

    :param model: model to be evaluated
    :param test_ds: test dataset to evaluate the model on
    :return: The accuracy, precision, recall and f1-score of the model
    """
    
    y_pred = []
    y_true = []

    for batch, labels in test_ds:
        batch_predictions = model.predict(batch)
        y_pred.append(batch_predictions)
        y_true.append(labels)

    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.concat(y_true, axis=0)

    tf.nn.softmax(y_pred)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1


def show_random_predictions(test_ds, labels, count=10):
    """
    The show_random_predictions function takes a dataset, labels, and count as arguments.
    The function shuffles the dataset and then iterates through it to display images with their true label
    and predicted label. The number of images displayed is determined by the count argument.

    :param test_ds: Pass the test dataset to the function
    :param labels: Map the integer label to a name
    :param count: Specify how many images to show
    """

    test_ds_unb = test_ds.unbatch()
    test_ds_unb.shuffle(120)

    i = 0
    for el, true_label in test_ds_unb:

        pred = model.predict(tf.expand_dims(el, 0))
        tf.nn.softmax(pred)
        pred_label = np.argmax(pred, axis=1)[0]

        el = np.array(el, dtype=np.uint8)
        if np.ndim(el) > 3:
            assert el.shape[0] == 1
            el = el[0]
        plt.imshow(el)
        plt.title(f"True label: {labels[true_label]}, Pred label: {labels[pred_label]}")
        plt.show()
        i += 1
        if i >= count:
            break


if __name__ == '__main__':
    train_ds, val_ds, test_ds = load_data()
    labels, num_classes = load_labels()
    model = prepare_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = train_model(model=model, train_ds=train_ds, val_ds=val_ds, epochs=18)
    model.save("model.keras")
    plot_learning_curves(history)
    accuracy, precision, recall, f1 = evaluate_model(model, test_ds)

    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 Score:", f1)

    show_random_predictions(test_ds, labels)
