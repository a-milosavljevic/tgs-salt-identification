"""
INCLUDE ONLY, DO NOT EXECUTE
"""
import numpy as np
from keras.utils import Sequence
from model import model_type


class DataAugmentation(Sequence):

    def __init__(self, x_set, y_set, batch_size, slide=False, scale=True, shift=True):
        self.x = x_set
        self.x_range = np.max(x_set) - np.min(x_set)
        self.y = y_set
        self.batch_size = batch_size
        self.slide = slide
        self.scale = scale
        self.shift = shift

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size].copy()
        if self.y is None:
            batch_y = None
        else:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size].copy()

        img_size = 128
        img_diff = 27
        img_half = 13
        if model_type == 'pspnet':
            img_size = 120
            img_diff = 19
            img_half = 9

        # Slide
        if self.slide:
            new_batch_x = np.zeros((self.batch_size, img_size, img_size, 1), dtype=np.float32)
            if batch_y is not None:
                new_batch_y = np.zeros((self.batch_size, img_size, img_size, 1), dtype=np.float32)
            for i in range(len(batch_x)):
                if np.random.rand() > 0.25:
                    sx, sy = np.random.randint(img_diff + 1, size=2)
                else:
                    sx, sy = img_half, img_half
                new_batch_x[i] = np.pad(batch_x[i], ((sy, img_diff - sy), (sx, img_diff - sx), (0, 0)), 'reflect')
                if batch_y is not None:
                    new_batch_y[i] = np.pad(batch_y[i], ((sy, img_diff - sy), (sx, img_diff - sx), (0, 0)), 'reflect')
            batch_x = new_batch_x
            if batch_y is not None:
                batch_y = new_batch_y

        # Intensity Scale
        if self.scale:
            batch_x[:, :, :, 0] *= np.random.uniform(0.8, 1.2)

        # Intensity Shift
        if self.shift:
            batch_x[:, :, :, 0] += np.random.uniform(-0.2, 0.2) * self.x_range

        if batch_y is None:
            return np.array(batch_x), None
        else:
            return np.array(batch_x), np.array(batch_y)
