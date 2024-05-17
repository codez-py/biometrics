import os
from glob import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Input
from tensorflow.keras.utils import plot_model


DATA_DIR = '/kaggle/input/casia-gei-gender/gei'

gt = pd.read_csv(f'{DATA_DIR}/GaitDatasetB-gender.csv', delimiter='\t')


def load_gei(f, t, angle):
    for person_id in range(f, t):
        ps = str(person_id).zfill(3)
        angle = str(angle).zfill(3)
        for file in sorted(glob(f'{DATA_DIR}/{ps}/nm-*/*{angle}.png')):
            gender = gt[gt['Subject ID'] == person_id].values[0][1]
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            img = img.reshape((*img.shape, 1))
            yield img, person_id, int(gender == 'F')


def get_features(dataset):
    features, _, labels = [*zip(*dataset)]
    features = tf.convert_to_tensor(np.array(features) / 255, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    return features, labels


def create_model():
    model = Sequential([
        Input((240, 240, 1)),
        Conv2D(8, (3,3), activation='relu'),
        MaxPool2D(),
        Conv2D(16, (3,3), activation='relu'),
        MaxPool2D(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPool2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(500, activation='relu'),
        Dense(40, activation='relu'),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


angles = [90, 72, 54, 36, 18]
epochs = 100
DATA_DIR = '/kaggle/input/casia-gei-gender/gei'

def train_all():
    for angle in angles:
        gait_dataset = list(load_gei(1, 75, angle))
        val_dataset = list(load_gei(75, 100, angle))

        features, labels = get_features(gait_dataset)
        val_features, val_labels = get_features(val_dataset)

        dataset  = tf.data.Dataset.from_tensors((features, labels))
        val_data  = tf.data.Dataset.from_tensors((val_features, val_labels))

        model = create_model()

        hist = model.fit(dataset, epochs=epochs, batch_size=10, validation_data=val_data)

        model.save(f'nm-{angle}.keras')

        np.save(f'nm-{angle}.hist', hist.history)
        

def test_all():
    comp = []
    for angle in angles:
        test_dataset = list(load_gei(100, 125, 90))
        test_features, test_labels = get_features(test_dataset)
        print(f'Model trained with angle: {angle}')
        model = load_model(f'nm-{angle}.keras')
        yprob = model.predict(test_features)
        ypred = yprob.flatten().round()

        classes = ['M', 'F']
        pd.DataFrame(metrics.confusion_matrix(test_labels, ypred), index=classes, columns=classes).to_csv(f'nm-{angle}-confusion.csv')
        with open(f'nm-{angle}-report.txt', 'w') as f:
            f.write(metrics.classification_report(test_labels, ypred))

        comp.append((
            angle,
            metrics.accuracy_score(test_labels, ypred),
            metrics.precision_score(test_labels, ypred),
            metrics.recall_score(test_labels, ypred),
            metrics.f1_score(test_labels, ypred),
        ))

    pd.DataFrame(comp, columns=['angle', 'accuracy', 'precision', 'recall', 'f1_score']).to_csv('comparison.csv', index=False)
    print('all saved sucessfully')


def plot_all():
    plot_model(create_model(), show_shapes=True, show_layer_activations=True, to_file='model.png')
    for angle in angles:
        hist = np.load(f'nm-{angle}.hist.npy', allow_pickle=True).item()
        train_loss = hist["loss"]
        val_loss = hist["val_loss"]
        train_acc = hist["accuracy"]
        val_acc = hist["val_accuracy"]
        xc = range(epochs)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(xc, train_loss, label="train")
        plt.plot(xc, val_loss, label="val")
        plt.xlabel("num of Epochs")
        plt.ylabel("loss")
        plt.title(f"train_loss vs val_loss {angle} degrees")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(xc, train_acc, label="train")
        plt.plot(xc, val_acc, label="val")
        plt.xlabel("num of Epochs")
        plt.ylabel("accuracy")
        plt.title(f"train_acc vs val_acc for {angle} degrees")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"nm-{angle}.png")

        
def save_zip():
    print('!zip bio_proj.zip *.csv *.txt *.png *.npy *.keras')
    

## uncomment and run
# train_all()
# test_all()
# plot_all()
