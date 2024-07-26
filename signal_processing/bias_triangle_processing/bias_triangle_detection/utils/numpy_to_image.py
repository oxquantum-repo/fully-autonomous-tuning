import cv2 as cv
import numpy as np

from .types import GrayImage


def img_norm(img: np.ndarray, max_val: int = 255) -> np.ndarray:
    img = img - img.min()
    img_max = img.max()
    if img.max() == 0:
        img_max = 1.0
    img = img / img_max * max_val
    img = np.uint8(img)
    stacked_img = np.stack((img,) * 3, axis=-1)

    return stacked_img


def to_gray(img: np.ndarray, max_val: int = 255) -> GrayImage:
    """Turn numpy array into grayscale image with max value of max_val."""
    img = img_norm(img, max_val=max_val)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
