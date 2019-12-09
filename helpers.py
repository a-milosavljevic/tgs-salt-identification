"""
INCLUDE ONLY, DO NOT EXECUTE
"""
import numpy as np
from keras import backend as K


def adjust_input(x):
    return x.astype(np.float32) / 255.0


def adjust_output(y):
    return y.astype(np.float32)


def iou(y_true, y_pred):
    y_true_ = K.round(K.flatten(y_true))
    y_pred_ = K.round(K.flatten(y_pred))
    intersection = K.minimum(y_true_, y_pred_)
    union = K.maximum(y_true_, y_pred_)
    score = (K.sum(intersection) + 1e-6) / (K.sum(union) + 1e-6)
    return score


def py_acc(y_true, y_pred, axis=0):
    y_true_ = np.round(y_true).astype(np.bool)
    y_pred_ = np.round(y_pred).astype(np.bool)
    ct = np.sum(np.logical_not(np.logical_xor(y_true_, y_pred_)).astype(np.float32), axis=axis)
    cf = np.sum(np.logical_xor(y_true_, y_pred_).astype(np.float32), axis=axis)
    m = ct / (ct + cf)
    return np.mean(m)


def py_iou(y_true, y_pred, axis=0):
    y_true_ = np.round(y_true)
    y_pred_ = np.round(y_pred)
    intersection = np.sum(np.minimum(y_true_, y_pred_), axis=axis).astype(np.float32)
    union = np.sum(np.maximum(y_true_, y_pred_), axis=axis).astype(np.float32)
    m = (intersection + 1e-6) / (union + 1e-6)
    return np.mean(m)


def run_length_enc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def fast_run_length_enc(img):
    """
    img: numpy array of shape (height, width), True - mask, False - background
    Returns run length as list
    """
    dots = np.where(img.T.flatten())[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
