"""
DOWNLOAD DATA FILES FROM https://www.kaggle.com/c/tgs-salt-identification-challenge/data
PLACE THEM TO data SUBFOLDER
UNPACK train.zip TO train FOLDER AND test.zip TO test FOLDER
THE STRUCTURE OF data FOLDER SHOULD BE LIKE THIS:
    data/depths.csv
    data/sample_submission.csv
    data/train.csv
    data/test/images/*.png
    data/train/images/*.png
    data/train/masks/*.png
"""
import numpy as np
import os
import cv2 as cv
import csv
from helpers import *


root_folder = os.getcwd()
data_folder = os.path.join(root_folder, 'data')
include_depths = False


########################################################################################################################
if include_depths:
    print("Read depths")
    depths_path = os.path.join(data_folder, 'depths.csv')
    depths = dict()
    with open(depths_path, 'rt') as f:
        reader = csv.reader(f)
        depths_list = list(reader)[1:]
        for row in depths_list:
            depths[row[0]] = int(row[1])
    min_depth = min(depths.values())
    max_depth = max(depths.values())


########################################################################################################################
print("Processing training images and masks")

train_images_folder = os.path.join(data_folder, 'train', 'images')
train_masks_folder = os.path.join(data_folder, 'train', 'masks')

train_images = [x[2] for x in os.walk(train_images_folder)][0]

if include_depths:
    train_x = np.array([], dtype=np.float32).reshape((0, 101, 101, 2))
else:
    train_x = np.array([], dtype=np.uint8).reshape((0, 101, 101, 1))
train_y = np.array([], dtype=np.bool).reshape((0, 101, 101, 1))

cnt = 0
for file_name in train_images:
    if include_depths:
        depth = depths[file_name[0:-4]]
        depth = 255 * (depth - min_depth) / (max_depth - min_depth)
    image_path = os.path.join(train_images_folder, file_name)
    mask_path = os.path.join(train_masks_folder, file_name)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        if image is None:
            print()
            print('Invalid image:', image_path)
        if mask is None:
            print()
            print('Invalid mask:', mask_path)
    else:
        if include_depths:
            image = np.append(image.reshape(101, 101, 1), np.full((101, 101, 1), depth, dtype=np.float32), axis=2)
            train_x = np.append(train_x, image.reshape(1, 101, 101, 2), axis=0)
        else:
            train_x = np.append(train_x, image.reshape(1, 101, 101, 1), axis=0)
        train_y = np.append(train_y, mask.reshape(1, 101, 101, 1) > 127, axis=0)

    cnt += 1
    if cnt % 100 == 0:
        print(".", end='', flush=True)
print('')
print(train_x.shape)


########################################################################################################################
print("Fixing order of training samples")

count_ones = train_y.astype(np.float32).reshape((train_y.shape[0], np.prod(train_y.shape[1:]))).sum(axis=1)
count_ones_ind = np.argsort(count_ones)

train_x_sorted = train_x[count_ones_ind]
train_y_sorted = train_y[count_ones_ind]

fixed_ind = []
for start_ind in range(5):
    for curr_ind in range(start_ind, len(count_ones_ind), 5):
        fixed_ind.append(curr_ind)

train_x_fixed = train_x_sorted[fixed_ind]
train_y_fixed = train_y_sorted[fixed_ind]


########################################################################################################################
print("Saving training images and masks")

if include_depths:
    path_train_x = os.path.join(root_folder, 'train_x_depth_fixed.npy')
else:
    path_train_x = os.path.join(root_folder, 'train_x_fixed.npy')
path_train_y = os.path.join(root_folder, 'train_y_fixed.npy')
np.save(path_train_x, train_x_fixed)
np.save(path_train_y, train_y_fixed)


########################################################################################################################
print("Loading sample_submission.csv")

submission_path = os.path.join(data_folder, 'sample_submission.csv')
with open(submission_path, 'rt') as f:
    reader = csv.reader(f)
    submission_records = list(reader)


########################################################################################################################
print("Processing test images")

if include_depths:
    test_x = np.array([], dtype=np.float32).reshape((0, 101, 101, 2))
else:
    test_x = np.array([], dtype=np.uint8).reshape((0, 101, 101, 1))
cnt = 0
for i in range(len(submission_records)-1):
    file_name = submission_records[i+1][0] + ".png"
    if include_depths:
        depth = depths[file_name[0:-4]]
        depth = 255 * (depth - min_depth) / (max_depth - min_depth)
    image_path = os.path.join(data_folder, 'test', 'images', file_name)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if include_depths:
        image = np.append(image.reshape(101, 101, 1), np.full((101, 101, 1), depth, dtype=np.float32), axis=2)
        test_x = np.append(test_x, image.reshape(1, 101, 101, 2), axis=0)
    else:
        test_x = np.append(test_x, image.reshape(1, 101, 101, 1), axis=0)

    cnt += 1
    if cnt % 100 == 0:
        print(".", end='', flush=True)
print('')
print(test_x.shape)


########################################################################################################################
print("Saving test images")

if include_depths:
    path_test_x = os.path.join(root_folder, 'test_x_depth.npy')
else:
    path_test_x = os.path.join(root_folder, 'test_x.npy')
np.save(path_test_x, test_x)

