"""
VISUALIZE ENSEMBLE OUTPUT FOR VALIDATION FILES
REQUIRE ENSEMBLE OF 5 NETWORKS IN tmp FOLDER
"""
import numpy as np
import tensorflow as tf
import keras
import os
import h5py
import cv2 as cv
import csv
from helpers import *
from model import *


root_folder = os.getcwd()

results_folder = os.path.join(root_folder, 'tmp')

output_folder = os.path.join(results_folder, 'visualization')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# Parameters
batch_size = 64
slide_augmentation = True


# Load data
path_train_x = os.path.join(root_folder, 'train_x_fixed.npy')
path_train_y = os.path.join(root_folder, 'train_y_fixed.npy')

data_x = np.load(path_train_x)
data_y = np.load(path_train_y)

partition_size = round(data_x.shape[0] / 5)


# Create evaluation.csv file
evaluation_file = 'evaluation.csv'
evaluation_path = os.path.join(output_folder, evaluation_file)
with open(evaluation_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'Acc', 'IoU'])

    # For each partition
    for partition in [1, 2, 3, 4, 5]:
        test_x = data_x[partition_size * (partition - 1):partition_size * partition].copy()
        test_y = data_y[partition_size * (partition - 1):partition_size * partition].copy()

        if slide_augmentation:
            if model_type == 'pspnet':
                test_x = np.pad(test_x, ((0, 0), (9, 10), (9, 10), (0, 0)), mode='reflect')
            else:
                test_x = np.pad(test_x, ((0, 0), (13, 14), (13, 14), (0, 0)), mode='reflect')

        # Load model
        model_path = os.path.join(results_folder, 'trained_model_{}.h5'.format(partition))
        model = keras.models.load_model(model_path,
                                        custom_objects={'iou': iou},
                                        compile=True)
        if partition == 1:
            model.summary()

        # Predict outputs
        y = model.predict(preprocessing(test_x), batch_size=batch_size)

        if slide_augmentation:
            if model_type == 'pspnet':
                test_x = test_x[:, 9:110, 9:110, :]
                y = y[:, 9:110, 9:110, :]
            else:
                test_x = test_x[:, 13:114, 13:114, :]
                y = y[:, 13:114, 13:114, :]

        gt_images = (255 * test_y).astype(np.uint8)
        out_images = (255 * np.round(y)).astype(np.uint8)

        for i in range(len(y)):
            base_name = str(partition_size * (partition - 1) + i).zfill(4)
            input_img = test_x[i]
            gt_img = gt_images[i]
            out_img = out_images[i]
            cv.imwrite(os.path.join(output_folder, base_name + '.png'), input_img)
            cv.imwrite(os.path.join(output_folder, base_name + '_gt.png'), gt_img)
            cv.imwrite(os.path.join(output_folder, base_name + '_out.png'), out_img)

            img_acc = py_acc(adjust_output(test_y[i:i + 1]), y[i:i + 1], axis=None)
            img_iou = py_iou(adjust_output(test_y[i:i+1]), y[i:i+1], axis=None)
            writer.writerow([partition, img_iou, img_acc])
            print('P{} -> Index: {} - IoU: {} - Acc: {}'.format(partition, i, img_iou, img_acc))

        del model
        model = None
