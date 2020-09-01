import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]

def process_image_file(filepath, top_percent, size):
    img = cv2.imread(filepath)
    img = crop_top(img, percent=top_percent)
    img = central_crop(img)
    img = cv2.resize(img, (size, size))
    return img

def get_classes(dataset):
    classes = []
    for line in dataset:
        classes.append(line.split()[2])
    return classes

def get_dataset(lines):
    dataset = []
    for line in lines:
        dataset.append(line)
    return dataset

class ProcessedDatasetGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            data_files,
            batch_size=1,
            input_shape=(224, 224),
            n_classes=3,
            num_channels=3,
            is_validation=True,
            mapping={
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
            },
            top_percent=0.08,
    ):
        'Initialization'
        self.datadir = data_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.is_validation = is_validation
        self.mapping = mapping
        self.n = 0
        self.top_percent = top_percent
        self.dataset = get_dataset(data_files)
        self.N = len(self.dataset)
        self.classes = get_classes(self.dataset)

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.n = 0

        yield batch_x, batch_y

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(
            (self.batch_size, *self.input_shape,
            self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.is_validation:
            folder = 'train'
        else:
            folder = 'test'
        for i in range(len(batch_files)):
            sample = batch_files[i].split()            

            x = process_image_file(os.path.join(self.datadir, folder, sample[1]),
                                   self.top_percent,
                                   self.input_shape[0])

            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y
        
        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)