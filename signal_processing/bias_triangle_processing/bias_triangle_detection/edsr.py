from typing import Callable, List
import warnings

from skimage.filters import gaussian
import cv2 as cv
import numpy as np
import xarray as xr

def get_origin(data: xr.DataArray) -> np.ndarray:
    """ Gets the origin wrt to the axes's value assuming linear steps """
    origin = [None, None]
    assert len(data.shape) == 2
    for i, key in enumerate(data.coords.keys()):
        axis = data.coords[key].values
        step = np.diff(axis)[0]
        origin[i] = np.round(-axis[0]/step).astype(int)
    # x and y are swapped
    return np.array(origin[::-1])

def get_parallel_axis_condition(axis: int=0, tolerance: float=10) -> Callable:
    """ Returns a condition that checks if the ellipse is parallel to the given axis. """
    def condition(ellipse: List) -> bool:
        """ Checks if the ellipse is parallel to the given axis. """
        ellipse_angle_with_x = ellipse[2] + 90*np.argmax(ellipse[1])
        angle_diff = ellipse_angle_with_x - axis*90
        return min(angle_diff % 180, (-angle_diff) % 180) < tolerance
    return condition

def get_origin_intersection_condition(origin: np.ndarray, tolerance: float=10) -> Callable:
    """ Returns a condition that checks if the ellipse intersects the given origin. """
    def condition(ellipse: List) -> bool:
        """ Checks if the ellipse intersects the given origin. """
        v = np.array(ellipse[0]) - origin
        angle_with_origin = np.arctan2(v[1], v[0]) * 180 / np.pi
        ellipse_angle_with_x = ellipse[2] + 90*np.argmax(ellipse[1])
        angle_diff = angle_with_origin - ellipse_angle_with_x
        # import pdb; pdb.set_trace()
        return min(angle_diff % 180, (-angle_diff) % 180) < tolerance
    return condition

def extract_edsr_spot(data: np.ndarray,
                      sigma: float = 1, 
                      circularity_thresh: float = 3,
                      relative_length: float = 0.3,
                      adaptive_window: int = 7,
                      condition: Callable = None) -> List[np.ndarray]:
    """ Filters the edsr data, finding the maximum spot within oblungated structures. """
    if condition is None:
        condition = lambda _: True
    smooth = gaussian(data, sigma=sigma)
    img = ((smooth - smooth.min()) / (smooth.max() - smooth.min()) * 255).astype(np.ubyte)
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, adaptive_window, 0)

    # Remove some small noise if any.
    erode = cv.erode(thresh,None)
    dilate = cv.dilate(erode,None)

    # Find contours
    contours, _ = cv.findContours(dilate,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    # filter contours
    candidates = []
    for contour in contours:
        try:
            ellipse = cv.fitEllipse(contour)
        except:
             continue
        length = np.max(ellipse[1])
        circ = length/np.max([0.001, np.min(ellipse[1])])
        if circ > circularity_thresh and length > np.min(data.shape)*relative_length and condition(ellipse):
            candidates.append(ellipse)
    
    mask = np.zeros_like(img)
    for candidate in candidates:
        cv.ellipse(mask, candidate, (1), thickness=cv.FILLED)
    filtered = smooth*mask
    max = np.argwhere(filtered == filtered.max())[0]
    if not candidates:
        warnings.warn("No peaks found. Assume no PSB.")
        return dilate, filtered, [None, None]
    return dilate, filtered, max
