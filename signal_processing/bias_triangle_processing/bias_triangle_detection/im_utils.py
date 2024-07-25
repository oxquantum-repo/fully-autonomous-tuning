from typing import Tuple

from skimage.filters import threshold_minimum
import numpy as np
import cv2 as cv


def img_norm(img: np.ndarray) -> np.ndarray:

    img = img - img.min()
    img_max = img.max()
    if img.max() == 0:
        img_max = 1.0
    img = img / img_max * 255
    img = np.uint8(img)
    stacked_img = np.stack((img,) * 3, axis=-1)

    return stacked_img

def img_resize(tr_img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    result = cv.resize(
        tr_img,
        shape,
        interpolation=cv.INTER_CUBIC,
    )
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    return gray

def img_resize_gray(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    tr_img = img_norm(img)
    return img_resize(tr_img, shape)

def img_res_gray(img: np.ndarray, res: int) -> np.ndarray:
    shape = (res * img.shape[1], res * img.shape[0])
    return img_resize_gray(img, shape)


def threshold_mad(im: np.ndarray, k: int = 4) -> np.ndarray:

    """ Median absolute deviation (MAD)
    Robust statistic (measure of dispersion), resilient to outliers.
    Effective for images with large & noisy background, with comparatively small signal (e.g. wide-shot).
    Uses a scale factor of 1.4868 to resemble a more robust std.
    Details: https://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    med = np.median(im)
    mad = np.median(np.abs(im.astype(np.float32) - med))

    return med + mad * k * 1.4826


def iou(groundtruth_mask: np.ndarray, pred_mask: np.ndarray) -> float:

    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect / union)

    return round(iou, 3)
