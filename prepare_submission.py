"""
PREPARES SUBMISSION CSV FILE
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
from data_augmentation import DataAugmentation


root_folder = os.getcwd()
results_folder = os.path.join(root_folder, 'tmp')
data_folder = os.path.join(root_folder, 'data')


# Parameters
batch_size = 64
partitions = 5
tta = True
slide_augmentation = True
model_prefix = 'trained_model'


########################################################################################################################
print("Loading sample_submission.csv")

submission_path = os.path.join(data_folder, 'sample_submission.csv')
with open(submission_path, 'rt', newline='') as f:
    reader = csv.reader(f)
    submission_records = list(reader)


########################################################################################################################
print("Loading test images")

img_size = 128
img_m1 = 13
img_m2 = 14
if model_type == 'pspnet':
    img_size = 120
    img_m1 = 9
    img_m2 = 10

path_test_x = os.path.join(root_folder, 'test_x.npy')
test_x = np.load(path_test_x)
if slide_augmentation:
    test_x = np.pad(test_x, ((0, 0), (img_m1, img_m2), (img_m1, img_m2), (0, 0)), mode='reflect')
print(test_x.shape)


x = preprocessing(test_x)
if tta:
    xf = x.copy()
    np.fliplr(xf)
y = None

for partition in range(1, partitions+1):
    predition_path = os.path.join(results_folder, '{}_{}_prediction.npy'.format(model_prefix, partition))
    if os.path.exists(predition_path):
        print("Loading prediction for model {}".format(partition))
        yp = np.load(predition_path)
    else:
        print("Loading pretrained model {}".format(partition))
        model_path = os.path.join(results_folder, '{}_{}.h5'.format(model_prefix, partition))
        model = keras.models.load_model(model_path, custom_objects={'iou': iou})

        print("Predicting outputs {}".format(partition))
        yp = model.predict(x, batch_size=batch_size)
        if tta:
            yp_tta = model.predict(xf, batch_size=batch_size)
            np.fliplr(yp_tta)
            yp += yp_tta
            yp /= 2

        del model

        if slide_augmentation:
            yp = yp[:, img_m1:(img_size-img_m2), img_m1:(img_size-img_m2), :]

        print("Saving prediction for model {}".format(partition))
        np.save(predition_path, yp)

    if yp.shape[1] == img_size:
        yp = yp[:, img_m1:(img_size-img_m2), img_m1:(img_size-img_m2), :]

    if y is None:
        y = yp
    else:
        y += yp

y /= partitions


########################################################################################################################
print("Fixing outputs for empty inputs")

for i in range(len(x)):
    ix = x[i]
    if np.min(ix) == np.max(ix):
        y[i].fill(0.0)


########################################################################################################################
print("Run-length encoding outputs")

cnt = 0
for i in range(len(submission_records)-1):
    mask = y[i] > 0.5
    submission_records[i+1][1] = fast_run_length_enc(mask)

    cnt += 1
    if cnt % 100 == 0:
        print(".", end='', flush=True)
print('')


########################################################################################################################
print("Saving submission.csv")

if tta:
    submission_file = 'submission_TTA.csv'
else:
    submission_file = 'submission.csv'
submission_path = os.path.join(results_folder, submission_file)
with open(submission_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(submission_records)
