import numpy as np
import cv2 as cv
from typing import List, Callable
from skimage.filters import gaussian
from bias_triangle_detection.edsr import get_parallel_axis_condition
import numpy as np
from skimage import filters


def is_1d_cotunneling(data: np.ndarray,
                      circularity_thresh: float = 3,
                      relative_length: float = 0.5,
                      tolerance: float = 20,
                      return_mask: bool = False) -> List:
    """ Filters the binary data, checking the presence of oblungated structures. """
    condition_x = get_parallel_axis_condition(axis=0, tolerance=tolerance)
    condition_y = get_parallel_axis_condition(axis=1, tolerance=tolerance)

    # Remove some small noise if any.
    erode = cv.erode(data,None)
    dilate = cv.dilate(erode,None)

    # Find contours
    contours, _ = cv.findContours(dilate,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    # filter contours
    candidates_x = []
    candidates_y = []
    for contour in contours:
        try:
            ellipse = cv.fitEllipse(contour)
        except:
             continue
        length = np.max(ellipse[1])
        circ = length/np.max([0.001, np.min(ellipse[1])])
        if circ > circularity_thresh and length > np.min(data.shape)*relative_length:
            if condition_x(ellipse):
                candidates_x.append(ellipse)
            if condition_y(ellipse):
                candidates_y.append(ellipse)
    labels = [len(candidates_x)>0, len(candidates_y)>0]
    if return_mask: 
        masks = []
        for candidates in [candidates_x, candidates_y]:
            mask = np.zeros_like(data)
            for candidate in candidates:
                cv.ellipse(mask, candidate, (1), thickness=cv.FILLED)
            masks.append(mask)
        
        return masks, labels
    return [None, None], labels

def adaptive_threshold(data: np.ndarray,
                       sigma: float = 1, 
                       inv = True) -> np.ndarray:
    data = data.copy()
    if inv:
        data *= -1
    data = data - np.min(data)*(1 - 0.001*np.sign(np.min(data)))
    data = np.log(data)
    q00, q30 = np.quantile(data, [0, 0.3])
    if q00 == q30:
        # Snake scan
        t = filters.threshold_otsu(data.to_numpy()[data > q00])
        return ((data.to_numpy() > t)*255).astype(np.ubyte)

    data = gaussian(data, sigma=sigma)
    data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.ubyte)
    return cv.threshold(data, 0, 255, cv.THRESH_TRIANGLE)[1]


def is_2d_cotunneling(data: np.ndarray,
                      grid_hole_proportion: float = 0.02,
                      return_mask: bool = False) -> List:
    """ Filters the binary data, checking the presence of grid structures. """
    n_components, labels = cv.connectedComponents(data)
    sum_of_components = np.zeros_like(data)
    is_cotunneling = False
    for label in range(1, n_components):
        mask = labels == label 
        pixels = np.argwhere(mask)
        # Check if the component touches all the edges of the image.
        if ( 0 in pixels[:, 0] and data.shape[0]-1 in pixels[:, 0] and
             0 in pixels[:, 1] and data.shape[1]-1 in pixels[:, 1]):
            sum_of_components += mask
            is_cotunneling = True
    # Look for holes in the data
    if not is_cotunneling:
        contours, hierachy = cv.findContours(data, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        if contours:
            for c, h in zip(contours, hierachy[0]):
                # h[3] is the parent contour index, if it is -1, it means that the contour has no parent.
                # If it is not -1 it is part of another contour, so it represents a hole.
                if h[3] != -1 and cv.contourArea(c) > data.size * grid_hole_proportion:
                    is_cotunneling = True

    if return_mask:
        return sum_of_components>0, is_cotunneling 
    return None, is_cotunneling


def get_cotunneling(data: np.ndarray,
                      sigma: float = 1, 
                      circularity_thresh: float = 3,
                      relative_length: float = 0.5,
                      tolerance: float = 20,
                      grid_hole_proportion: float = 0.02,
                      inv: bool = True) -> List: 
    """ Returns presence of cotunneling for wide shot data, finding the maximum spot within oblungated structures. """
    thresh = adaptive_threshold(data, sigma=sigma, inv=inv)
    cotunneling_x, cotunneling_y = is_1d_cotunneling(thresh, relative_length=relative_length, circularity_thresh=circularity_thresh, tolerance=tolerance)[1]
    cotunneling_x_y = is_2d_cotunneling(thresh, grid_hole_proportion)[1]
    return [cotunneling_x or cotunneling_x_y, cotunneling_y or cotunneling_x_y] 