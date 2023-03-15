import cv2
import numpy as np
from imageio import imread

def ReadImage(path, shape):
    img = np.asarray(imread(path, pilmode='RGB'))
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: 
        img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_CUBIC)

    out = img.astype(np.float32) / 255
    return out