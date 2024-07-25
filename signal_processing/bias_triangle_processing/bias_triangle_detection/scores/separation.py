from typing import Callable, List, Optional, Tuple
from scipy.signal import find_peaks

import scipy.ndimage
import numpy as np
import xarray as xr

from bias_triangle_detection.btriangle_properties import detect_base_alt
from bias_triangle_detection.scores.base import ScoreResult
from bias_triangle_detection.switches.characterisation import get_masks_as_xr
from bias_triangle_detection.utils.contours_and_masks import get_individual_components
from bias_triangle_detection.utils.numpy_to_image import to_gray
from bias_triangle_detection.utils.xarray_and_qcodes import xr_to_mask
from bias_triangle_detection.switches.fastai_model_wrapper import FastAISwitchModel
from bias_triangle_detection.utils.types import PathLike

DEFAULT_SCORE = 1.0

def _score_separation(gray_orig: np.ndarray, masks: np.ndarray, direction: str = 'down'):
    base, corner_pts, _ = detect_base_alt(gray_orig, masks, direction)
    # Gets the short_side of the quadrilateral
    short_side = list(set(corner_pts) - set(tuple(b) for b in base))
    if len(short_side) != 2:
        return 1, None, None, None
    # sort the short_side in such a way that the segments from short_side[i] to base[i] correspond to the two sides of the quadrilateral
    short_side.sort(key = lambda p: np.linalg.norm(p - base[0]))

    # Loops through all the segments from the short_side to the base and gets the average value of the pixels in the segment
    n_steps = 100
    z = []
    for line in np.linspace(np.ravel(short_side), np.ravel(base), n_steps):
        z.append(get_values_from_segment(gray_orig, line).mean())
    z = np.array(z)

    # It should find two peaks and a dip in the middle, corresponding to the separation location
    peaks, _ = find_peaks(z)
    score = DEFAULT_SCORE
    min_loc = None
    if len(peaks) == 2:
        min_loc = np.argmin(z[peaks[0]:peaks[1]+1]) + peaks[0]
        # we score by the relative difference wrt the lowest peak
        score = min(z[peaks]/z[min_loc])
    return score, z, peaks, min_loc

def get_values_from_segment(image: np.ndarray, segment: List, n_steps = 100) -> np.ndarray:
    """ Get values from a segment in an image"""
    x0, y0, x1, y1 = segment
    x, y = np.linspace(x0, x1, n_steps), np.linspace(y0, y1, n_steps)
    zi = scipy.ndimage.map_coordinates(image, np.vstack((y,x)))
    return zi

def get_above_current_mask(readout_data: xr.DataArray, threshold: float = 2e-10) -> np.ndarray:
    assert readout_data.units == "A", "The readout data should be in units of Ampere"
    background = readout_data.median().item()
    return np.abs(readout_data.to_numpy() - background) > threshold 

def filter_by_current(readout_data: xr.DataArray, masks: List[np.ndarray], threshold: float = 2e-10) -> Tuple[List[np.ndarray], np.ndarray]:
    above_current_mask = get_above_current_mask(readout_data, threshold)
    return [m for m in masks if not np.any(above_current_mask[m!=0])], above_current_mask

def score_separation(
    readout_data: xr.DataArray,
    *,
    slow_axis: str = "rp",
    threshold_method: str = "default",
    direction: str = "down",
    score_statistic: Callable[[List[float]], ScoreResult] = np.max,
    switch_model: Optional[FastAISwitchModel] = None,
    triangles_mask: Optional[xr.DataArray] = None,
) -> ScoreResult:
    if triangles_mask is None:
        assert switch_model is not None, "If triangles_mask is not provided then the switch_model should be provided."
        triangles_mask = get_masks_as_xr(readout_data, slow_axis, threshold_method, switch_model)[0][0]
    masks = get_individual_components(xr_to_mask(triangles_mask))[0]
    masks = filter_by_current(readout_data, masks)[0]
    gray_data = -to_gray(readout_data)
    score_per_component = [_score_separation(gray_data, m, direction)[0] for m in masks] + [DEFAULT_SCORE]
    return score_statistic(score_per_component)