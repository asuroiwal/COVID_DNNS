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

def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 480 or size[1] > 480:
        print(img.shape, size, ratio)

    img = cv2.resize(img, size)
    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT,
                             (0, 0, 0))

    if img.shape[0] != 480 or img.shape[1] != 480:
        raise ValueError(img.shape, size)
    return img

_augmentation_transform = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,
)

def apply_augmentation(img):
    img = random_ratio_resize(img)
    img = _augmentation_transform.random_transform(img)
    return img

'''def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files'''

def get_classes(dataset):
    classes = []
    for line in dataset:
        classes.append(line.split()[2])
    return classes


def get_datasets(lines):
    datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    dataset = []
    for line in lines:
        datasets[line.split()[2]].append(line)
        dataset.append(line)
    return dataset, datasets

class BalanceCovidDataset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir,
            data_files,
            is_training=True,
            is_validation=False,
            is_testing=False,
            batch_size=8,
            input_shape=(224, 224),
            n_classes=3,
            num_channels=3,
            mapping={
                'normal': 0,
                'pneumonia': 1,
                'COVID-19': 2
            },
            shuffle=True,
            augmentation=apply_augmentation,
            covid_percent=0.3,
            top_percent=0.08
    ):
        'Initialization'
        self.datadir = data_dir
        self.is_training = is_training
        self.is_testing= is_testing
        self.is_validation = is_validation
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.covid_percent = covid_percent
        self.n = 0
        self.augmentation = augmentation
        self.top_percent = top_percent
        self.dataset, self.datasets = get_datasets(data_files)
        datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in self.dataset:
            datasets[l.split()[2]].append(l)
        self.datasets = [
            datasets['normal'] + datasets['pneumonia'],
            datasets['COVID-19'],
        ]
        print(len(self.datasets[0]), len(self.datasets[1]))
        self.N = len(self.dataset)
        self.classes = get_classes(self.dataset)

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        yield batch_x, batch_y

    def __len__(self):
        if self.is_testing:
            return int(np.ceil(len(self.datasets) / float(self.batch_size)))
        if self.is_validation:
            return int(np.ceil(len(self.datasets) / float(self.batch_size)))
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        if self.is_training or self.is_validation:
            batch_x, batch_y = np.zeros(
                (self.batch_size, *self.input_shape,
                 self.num_channels)), np.zeros(self.batch_size)

            batch_files = self.datasets[0][idx * self.batch_size:(idx + 1) *
                                           self.batch_size]

            # upsample covid cases
            covid_size = max(int(len(batch_files) * self.covid_percent), 1)
            covid_inds = np.random.choice(np.arange(len(batch_files)),
                                          size=covid_size,
                                          replace=False)
            covid_files = np.random.choice(self.datasets[1],
                                           size=covid_size,
                                           replace=False)
            for i in range(covid_size):
                batch_files[covid_inds[i]] = covid_files[i]

            for i in range(len(batch_files)):
                sample = batch_files[i].split()

                if self.is_training:
                    folder = 'train'
                elif self.is_validation:
                    folder = 'train'
                elif self.is_testing:
                    folder = 'test'

                x = process_image_file(os.path.join(self.datadir, folder, sample[1]),
                                       self.top_percent,
                                       self.input_shape[0])

                if self.is_training and hasattr(self, 'augmentation'):
                    x = self.augmentation(x)

                x = x.astype('float32') / 255.0
                y = self.mapping[sample[2]]

                batch_x[i] = x
                batch_y[i] = y
        else:
            batch_x, batch_y = np.zeros(
                (self.batch_size, *self.input_shape,
                 self.num_channels)), np.zeros(self.batch_size)

            batch_files = self.dataset[idx * self.batch_size:(idx + 1) *
                                           self.batch_size]

            for i in range(len(batch_files)):
                sample = batch_files[i].split()
                folder = 'test'

                x = process_image_file(os.path.join(self.datadir, folder, sample[1]),
                                       self.top_percent,
                                       self.input_shape[0])

                x = x.astype('float32') / 255.0
                y = self.mapping[sample[2]]

                batch_x[i] = x
                batch_y[i] = y
        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)