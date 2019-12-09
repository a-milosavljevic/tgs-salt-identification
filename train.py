"""
BE SURE TO RUN prepare_data.py FIRST
TO SETUP MODEL SEE model.py
"""
import numpy as np
import tensorflow as tf
import keras
import os
import h5py
import cv2 as cv
from sklearn.utils import shuffle
from helpers import *
from model import *
from data_augmentation import DataAugmentation


print("Tensorflow version: " + tf.__version__)
print("Keras version: " + keras.__version__)


##################################################################################
# NETWORK PARAMETERS
##################################################################################

initial_filters = 32
dropout = 0.2
batch_size = 32
max_epochs = 1000
data_augmentation = True
slide_augmentation = True
double_size = True


##################################################################################
# LOAD DATA
##################################################################################

root_folder = os.getcwd()
tmp_folder = os.path.join(root_folder, 'tmp')
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

path_train_x = os.path.join(root_folder, 'train_x_fixed.npy')
path_train_y = os.path.join(root_folder, 'train_y_fixed.npy')

data_x = np.load(path_train_x)
data_y = np.load(path_train_y)

partition_size = round(data_x.shape[0] / 5)


for partition in [1, 2, 3, 4, 5]:

    print("")
    print("************************************* PARTITION {} *************************************".format(partition))
    print("")

    test_x = data_x[partition_size * (partition - 1):partition_size * partition].copy()
    test_y = data_y[partition_size * (partition - 1):partition_size * partition].copy()
    train_x1 = data_x[0:partition_size * (partition - 1)].copy()
    train_y1 = data_y[0:partition_size * (partition - 1)].copy()
    train_x2 = data_x[partition_size * partition:].copy()
    train_y2 = data_y[partition_size * partition:].copy()
    train_x = np.append(train_x1, train_x2, axis=0)
    train_y = np.append(train_y1, train_y2, axis=0)

    # Data augmentation (first level)
    train_x = np.append(train_x, [np.fliplr(x) for x in train_x], axis=0)
    train_y = np.append(train_y, [np.fliplr(x) for x in train_y], axis=0)
    test_x = np.append(test_x, [np.fliplr(x) for x in test_x], axis=0)
    test_y = np.append(test_y, [np.fliplr(x) for x in test_y], axis=0)

    print(test_x.shape)
    print(test_y.shape)
    print(train_x.shape)
    print(train_y.shape)

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y, random_state=0)

    if data_augmentation and slide_augmentation:
        if model_type == 'pspnet':
            test_x = np.pad(test_x, ((0, 0), (9, 10), (9, 10), (0, 0)), mode='reflect')
            test_y = np.pad(test_y, ((0, 0), (9, 10), (9, 10), (0, 0)), mode='reflect')
        else:
            test_x = np.pad(test_x, ((0, 0), (13, 14), (13, 14), (0, 0)), mode='reflect')
            test_y = np.pad(test_y, ((0, 0), (13, 14), (13, 14), (0, 0)), mode='reflect')
    else:
        slide_augmentation = False

    ##################################################################################
    # MODEL DEFINITION
    ##################################################################################

    model = create_model(double_size=double_size,
                         slide_augmentation=slide_augmentation,
                         trainable_encoder=True,
                         n=initial_filters,
                         dropout=dropout)

    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.binary_crossentropy,
                  metrics=['acc', iou])

    if partition == 1:
        model.summary()

    ##################################################################################
    # TRAINING
    ##################################################################################

    model_path = os.path.join(tmp_folder, 'trained_model_{}.h5'.format(partition))
    training_log_path = os.path.join(tmp_folder, 'training_log_{}.csv'.format(partition))
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True,
                                                   monitor='val_acc', mode='max')
    csv_logger = keras.callbacks.CSVLogger(training_log_path, separator=',', append=False)
    early_stopping = keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True, verbose=1,
                                                   monitor='val_acc', mode='max')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-7, verbose=1,
                                                  monitor='val_acc', mode='max')

    if not data_augmentation:
        history = model.fit(preprocessing(train_x),
                            adjust_output(train_y),
                            batch_size=batch_size,
                            epochs=max_epochs,
                            validation_data=(preprocessing(test_x), adjust_output(test_y)),
                            shuffle=True,
                            callbacks=[checkpointer, csv_logger, early_stopping, reduce_lr],
                            verbose=2)
    else:
        data_gen = DataAugmentation(preprocessing(train_x),
                                    adjust_output(train_y),
                                    batch_size,
                                    slide=slide_augmentation,
                                    scale=True,
                                    shift=True)
        history = model.fit_generator(data_gen,
                                      epochs=max_epochs,
                                      validation_data=(preprocessing(test_x), adjust_output(test_y)),
                                      shuffle=True,
                                      callbacks=[checkpointer, csv_logger, early_stopping, reduce_lr],
                                      verbose=2)

    model.save(model_path, include_optimizer=False)

    # Delete the model
    del model
    model = None
