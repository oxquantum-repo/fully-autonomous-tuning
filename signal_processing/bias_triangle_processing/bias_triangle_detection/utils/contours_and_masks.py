from typing import Callable, List, Tuple

import cv2 as cv
import numpy as np

from .numpy_to_image import to_gray
from .types import Contour, GrayImage, Mask


def get_filtered_contours(
    mask: Mask, condition: Callable[[Contour], bool]
) -> List[Contour]:
    """Find contours in mask that satisfy condition."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return [contour for contour in contours if condition(contour)]


def contours_to_mask(contours: List[Contour], shape: Tuple[int, int]) -> Mask:
    """Draw contours on a canvas."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv.drawContours(mask, contours, -1, 1, -1)
    return mask


def mask_image(_img: GrayImage, mask: Mask) -> GrayImage:
    """Only for visualization purposes. This function creates an image with _img as a faint background (max val 126) with the mask drawn on top.
    These values are chosen for contrast."""
    max_val_for_background_image = 126
    img = to_gray(_img, max_val_for_background_image)
    img[mask > 0] = 255
    return img


def get_individual_components(masks: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    ret, labels = cv.connectedComponents(masks)
    components = []

    for label in range(1, ret):
        mask_edge = np.uint8(np.zeros_like(labels))
        mask_edge[labels == label] = 255
        components.append(mask_edge)
    return components, labels
