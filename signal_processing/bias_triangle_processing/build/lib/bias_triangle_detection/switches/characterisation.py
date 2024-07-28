from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage.filters import gaussian
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from bias_triangle_detection.btriangle_detection import triangle_segmentation_alg
from bias_triangle_detection.btriangle_properties import detect_base_alt
from bias_triangle_detection.im_utils import threshold_mad
from bias_triangle_detection.utils.contours_and_masks import (
    contours_to_mask,
    get_filtered_contours,
    get_individual_components,
    mask_image,
)
from bias_triangle_detection.utils.numpy_to_image import to_gray
from bias_triangle_detection.utils.types import Contour, GrayImage, Mask
from bias_triangle_detection.utils.xarray_and_qcodes import xarrays_to_visualize


class SwitchModel(Protocol):
    def __call__(self, image: GrayImage) -> bool:
        pass
    

def get_outlier_threshold(
    gray: GrayImage, extreme_pct: float = 0.02, n_std: float = 6
) -> np.uint8:
    """Outlier threshold based on number of std from median.

    Args:
        gray: grayscale image
        extreme_pct: percentile of extreme values to ignore
        n_std: number of std to consider outlier

    Returns:
        threshold value
    """
    pixel_values = gray.flatten()
    # std only from non extreme values
    lq, hq = np.quantile(pixel_values, [extreme_pct / 2, 1 - extreme_pct / 2])
    values_in_between = pixel_values[(pixel_values > lq) & (pixel_values < hq)]
    std = np.std(values_in_between)
    iqr_outliers = pixel_values[(pixel_values - np.median(pixel_values)) > n_std * std]
    if len(iqr_outliers) == 0:
        return 0
    return min(iqr_outliers)


def get_horizontal_segments_mask(horizontal: Mask, horizontal_size: int) -> Mask:
    """
    Given a mask `horizontal`, get number of horizontal segments matching a segment of size `horizontal_size`.

    Args:
        horizontal: the mask to process
        horizontal_size: the size of the segments to find

    Returns:
        the mask containing the segments found
    """

    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    return horizontal


def contour_is_line(contour: Contour, line_size: int = 0) -> bool:
    """Check if a contour is a line.

    Args:
        contour: The contour to check.
        line_size: The minimum number of points to consider the contour as a line.

    Returns:
        True if the contour is a line, False otherwise.
    """
    return cv.contourArea(contour) == 0 and len(contour) >= line_size


def get_horizontal_switches_masks(mask: Mask, horizontal_size: int) -> List[Mask]:
    """Get masks of horizontal switches from mask

    Args:
        mask: binary image
        horizontal_size: width of horizontal switches

    Returns:
        list of masks of horizontal switches
    """
    horizontal_segments_mask = get_horizontal_segments_mask(mask, horizontal_size)
    contours = get_filtered_contours(
        horizontal_segments_mask, partial(contour_is_line, line_size=horizontal_size)
    )
    return [contours_to_mask([c], mask.shape) for c in contours]


def get_diff(img: np.ndarray, axis: int = 0, sigma: float = 0.6) -> np.ndarray:
    """Get the difference along an axis.

    Args:
        img: the image to get difference from
        axis: the axis to get difference along
        sigma: the sigma for the Gaussian filter

    Returns:
        the difference image
    """
    assert axis in [0, 1], "axis must be 0 or 1"
    # get the gradient along the fast axis
    img = gaussian(img, sigma)
    # add row that will be removed with diff
    row_column = img[0][None, :] if axis == 0 else img[:, 0][:, None]
    return np.diff(img, axis=axis, prepend=row_column)


def get_outlier_mask(
    gray: GrayImage, threshold_getter: Callable[[GrayImage], float]
) -> Mask:
    """Get the mask of outliers.

    Args:
        gray: the gray image.
        threshold_getter: function to get the threshold.

    Returns:
        The mask of the outliers.
    """
    threshold = threshold_getter(gray)
    # this sort get's rid of ambigous cases when current is +/-
    return sorted(
        [
            cv.threshold(_gray, threshold, 1, cv.THRESH_BINARY)[1]
            for _gray in [gray, cv.bitwise_not(gray)]
        ],
        key=lambda mask: np.sum(mask),
    )[0]


def get_switches_masks(
    mask: Mask, min_switch_size: int = 3, max_switch_size: int = 10
) -> List[Mask]:
    """Get switches masks for a given mask.

    Args:
        mask: The mask for which the switches masks are computed.
        min_switch_size: The minimum switch size (in pixels).
        max_switch_size: The maximum switch size (in pixels).

    Returns:
        A list of switches masks.
    """
    # get individual switches masks for each switch size.
    return [
        m
        for switch_size in range(min_switch_size, max_switch_size + 1)
        for m in get_horizontal_switches_masks(mask, switch_size)
        if m.sum() > 0
    ]


def get_switch_characterization(
    img: np.ndarray,
    threshold_getter: Callable[[GrayImage], float],
    axis: int = 0,
    sigma: float = 0.6,
    min_switch_size: int = 3,
    max_switch_size: int = 10,
    switch_model: Optional[SwitchModel] = None,
) -> Tuple[Mask, Mask, Mask, Mask, Mask, GrayImage, np.ndarray, np.ndarray]:
    """Obtain mask of candidate switch free triangles and context masks.

    Args:
        img (np.ndarray): original image
        threshold_getter (Callable[[GrayImage], float]): method to get the threshold from image of gradients
        axis (int, optional): axis perpenticular to switches. Defaults to 0.
        sigma (float, optional): sigma for gaussian smoothing of image. Defaults to 0.6.
        min_switch_size (int, optional): min expected size of switches. Defaults to 3.
        max_switch_size (int, optional): max expected size of switches. Defaults to 10.

    Returns:
        Tuple[Mask, ...]: mask of good candidates, mask of all candidates, mask of corner points, mask of switches, mask of outliers, gray image of gradients
    """
    diff = get_diff(img, axis=axis, sigma=sigma)
    gray_diff = to_gray(diff)
    outlier_mask = get_outlier_mask(gray_diff, threshold_getter)
    switches_masks = get_switches_masks(outlier_mask, min_switch_size, max_switch_size)
    # TODO: remove `candidate_has_switch` filter and adjust all downstream code accordingly
    (
        candidate_masks,
        corner_pts_mask,
        candidate_mask,
        crops,
        crops_idcs
    ) = extract_candidate_triangle_masks(img, switch_model=switch_model)
    filtered_candidates = [
        candidate_mask
        for candidate_mask in candidate_masks
        if not candidate_has_switch(candidate_mask, switches_masks)
    ]
    filtered_candidates_mask = (
        np.sum(np.array(filtered_candidates), axis=0) > 0
        if len(filtered_candidates) > 0
        else np.zeros_like(img)
    )
    switch_mask = (
        np.sum(np.array(switches_masks), axis=0) > 0
        if len(switches_masks) > 0
        else np.zeros_like(img)
    )

    
    return (
        filtered_candidates_mask,
        candidate_mask,
        corner_pts_mask,
        switch_mask,
        outlier_mask,
        gray_diff,
        np.array(crops),
        np.array(crops_idcs)
    )


def candidate_has_switch(
    candidate_mask: Mask, switches_masks: List[Mask], erode_size_edge: int = 1
) -> bool:
    """
    Check if a candidate mask has a switch in it.

    Args:
        candidate_mask: The candidate mask to check for switches.
        switches_masks: All the switches masks to check in.

    Returns:
        True if the candidate mask has a switch in it, False otherwise.
    """
    # we obtain the start and end of the candidate mask along the slow axis
    start_mask = np.nonzero(candidate_mask)[0][0]
    end_mask = np.nonzero(candidate_mask)[0][-1]
    # we erode the candidate mask at the edges to avoid detecting switches at the edges
    candidate_mask_eroded = candidate_mask.copy()
    candidate_mask_eroded[: start_mask + erode_size_edge, :] = 0
    candidate_mask_eroded[end_mask - erode_size_edge :, :] = 0
    return any(
        [
            mask_in_mask(candidate_mask_eroded, switch_mask)
            for switch_mask in switches_masks
        ]
    )


def make_threshold_getter(threshold_method: str = "default") -> Callable:
    if threshold_method == "mad":
        outlier_kwargs = {"k": 4}
        return partial(threshold_mad, **outlier_kwargs)
    if threshold_method == "default":
        outlier_kwargs = {
            "extreme_pct": 0.02,
            "n_std": 5,
        }
        return partial(get_outlier_threshold, **outlier_kwargs)
    raise ValueError(f"threshold_method {threshold_method} not recognized")


def get_masks_as_xr(
    dataset: xr.DataArray,
    slow_axis: str = "rp",
    threshold_method: str = "default",
    switch_model: Optional[SwitchModel] = None,
    **switches_kwargs: Any,
) -> Tuple[Tuple[xr.DataArray], List[float]]:
    """Extract masks as xr.Datasets and the axes dimensions.

    Args:
        dataset (xr.Dataset): xarray of scan
        threshold_method (str, optional): outlier threshold method. Defaults to "default".
        slow_axis (str, optional): axis perpendicular to switches. Defaults to "rp".

    Returns:
        Tuple[Tuple[xr.Dataset], List[float]]: mask of good candidates, mask of all candidates, mask of corner points, mask of switches, mask of outliers, gray image of gradients and the axes dims.
    """
    assert len(dataset.dims) == 2, "dataset must have 2 dimensions"
    axes_dims = sorted(
        dataset.dims, key=lambda dim: slow_axis.lower() in dim.lower(), reverse=True
    )
    switches_kwargs["threshold_getter"] = make_threshold_getter(threshold_method)
    # we make model explicit on function signature for readability
    switches_kwargs["switch_model"] = switch_model
    (
        filtered_candidates_mask,
        candidate_mask,
        corner_pts_mask,
        switch_mask,
        outlier_mask,
        gray_diff,
        crops,
        crops_idcs
    ) = xr.apply_ufunc(
        get_switch_characterization,
        dataset,
        input_core_dims=[axes_dims],
        output_core_dims=[axes_dims] * 6
        + [["crops", *[f"{dim}_crop" for dim in axes_dims]]]
        + [["crops", "idcs"]],
        kwargs=switches_kwargs,
    )
    return (
        filtered_candidates_mask,
        candidate_mask,
        corner_pts_mask,
        switch_mask,
        outlier_mask,
        gray_diff,
        crops,
        crops_idcs
    ), axes_dims


def create_viz_plot(
    original: xr.DataArray,
    filtered_candidates_mask: xr.DataArray,
    candidate_mask: xr.DataArray,
    corner_pts_mask: xr.DataArray,
    switch_mask: xr.DataArray,
    outlier_mask: xr.DataArray,
    gray_diff: xr.DataArray,
    slow_axis: str,
    fast_axis: str,
) -> plt.Figure:
    """Generate plot for debugging."""
    switch_mask_with_diff = xr.apply_ufunc(mask_image, gray_diff, switch_mask)
    outlier_mask_with_diff = xr.apply_ufunc(mask_image, gray_diff, outlier_mask)
    viz_image_masks = [
        switch_mask,
        filtered_candidates_mask,
        candidate_mask,
        corner_pts_mask,
    ]
    viz_image = overlay_masks(viz_image_masks[::-1])

    # create visualization axis

    viz = xarrays_to_visualize(
        {
            "original": xr.apply_ufunc(to_gray, original),
            "switch_detection": viz_image,
            "corner_pts": xr.apply_ufunc(
                lambda m: to_gray(m.astype(np.uint8)), corner_pts_mask
            ),
            "candidates_mask": xr.apply_ufunc(
                lambda m: to_gray(m.astype(np.uint8)), candidate_mask
            ),
            "filtered_candidates_mask": xr.apply_ufunc(
                lambda m: to_gray(m.astype(np.uint8)), filtered_candidates_mask
            ),
            "diff": gray_diff,
            "outlier_mask": outlier_mask_with_diff,
            "switch_mask": switch_mask_with_diff,
        }
    )

    plots = viz.plot(x=slow_axis, y=fast_axis, col="viz")
    return plots.fig


def extract_candidate_triangle_masks(
    readout_data: np.ndarray,
    switch_model: Optional[SwitchModel] = None,
    res_h: int = 4,
    min_area_f: float = 0.001,
    thr_method: str = "triangle",
    denoising: bool = True,
    allow_MET: bool = False,
    direction: str = "down",
    inv: bool = True,
) -> Tuple[List[Mask], Mask, Mask, List[GrayImage], List[xr.DataArray]]:
    """Extract mask of candidate triangles by clustering on triangle shape"""
    min_area = min_area_f * readout_data.shape[0] * readout_data.shape[1] * (res_h**2)
    gray_data, _, masks = triangle_segmentation_alg(
        readout_data,
        res=res_h,
        min_area=min_area,
        thr_method=thr_method,
        denoising=denoising,
        allow_MET=allow_MET,
        direction=direction,
        inv=inv,
    )
    individual_components = get_individual_components(masks)[0]
    contours = []
    for component_mask in individual_components:
        _, corner_pts, _ = detect_base_alt(gray_data, component_mask, direction)
        contours.append(np.array(corner_pts).reshape(-1, 1, 2))

    candidate_masks, crops, crops_idcs = get_candidate_masks_from_model(
        switch_model, contours, gray_data
    )
    corner_pts_mask = contours_to_mask(contours, gray_data.shape)
    if res_h != 1:
        masks = cv.resize(masks, readout_data.shape, interpolation=cv.INTER_CUBIC)
        corner_pts_mask = cv.resize(
            corner_pts_mask, readout_data.shape, interpolation=cv.INTER_CUBIC
        )
        candidate_masks = [
            cv.resize(m, readout_data.shape, interpolation=cv.INTER_CUBIC)
            for m in candidate_masks
        ]
        candidate_mask = (
            np.sum(np.array(candidate_masks), axis=0) > 0
            if len(candidate_masks) > 0
            else np.zeros_like(readout_data)
        )
    return candidate_masks, corner_pts_mask, candidate_mask, crops, crops_idcs


def mask_in_mask(mask: Mask, mask_to_check: Mask) -> bool:
    """Check if mask_to_check is in mask"""
    if mask_to_check.sum() == 0:
        return False
    assert mask.shape == mask_to_check.shape, "masks must have same shape"
    intersection = np.logical_and(mask, mask_to_check)
    return (intersection == mask_to_check).all()


def get_candidate_labels(
    X: np.ndarray, min_number_of_points: int = 2
) -> Tuple[np.ndarray]:
    """Simple mean field classifier."""
    quantile = 0.2

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=quantile)

    EPSILON = 1e-5
    algorithm = cluster.MeanShift(bandwidth=bandwidth + EPSILON, bin_seeding=True)

    algorithm.fit(X)
    return post_process_labels(algorithm, X, min_number_of_points=min_number_of_points)


def post_process_labels(
    algorithm: cluster.MeanShift, X: np.ndarray, min_number_of_points: int = 2
) -> np.ndarray:
    # Post process labels from meanshift so that:
    # 1. dont provide single candidates
    # 2. assume that clusters of size 1 are neglected when there is a unique 'big' cluster
    # 3. if there are multiple clusters of size >1, take the biggest
    y_pred = algorithm.labels_.astype(int)
    labels, label_counts = np.unique(y_pred, return_counts=True)
    labels_with_at_least_min_number_of_points = labels[
        label_counts >= min_number_of_points
    ]
    if len(labels_with_at_least_min_number_of_points) == 0:
        return np.array([])
    if len(labels_with_at_least_min_number_of_points) == 1:
        return np.arange(len(y_pred))
    if len(labels_with_at_least_min_number_of_points) >= 2:
        lvl = lowest_variance_label(
            X, y_pred, labels_with_at_least_min_number_of_points
        )
        return np.arange(len(y_pred))[y_pred == 0]


def get_candidate_masks_from_model(
    switch_model: Union[SwitchModel, None],
    contours: List[Contour],
    gray_data: GrayImage,
) -> Tuple[List[Mask], List[GrayImage], List[xr.DataArray]]:
    # TODO: update docstring
    """Get mask of candidate triangles by clustering on triangle coordinates with respect to centroid.

    Args:
        contours (List[Contour]): contours of triangles
        shape (Tuple[int, int]): shape of image

    Returns:
        List[Mask]: list of masks of candidate triangles
        crops: np.array of (n_crops, crop_width, crop_height)
        crop_idcs: list of bounding box pixel corners for each crop 
    """
    shape = gray_data.shape
    contours = [
        c for c in contours if not mask_at_boundary(contours_to_mask([c], shape))
    ]
    contours = [c for c in contours if len(c) == 4]
    candidate_masks = [contours_to_mask([c], shape) for c in contours]
    crops = []
    crops_idcs = []
    for mask in candidate_masks:
        candidates = crop_from_mask(mask, gray_data)
        crops.append(candidates[0])
        crops_idcs.append(candidates[1])
    if len(crops) == 0:
        crops = np.zeros((0, 25, 25))
        crops_idcs = np.zeros((0,0))
    if switch_model is None:
        return candidate_masks, crops, crops_idcs
    return (
        [
            mask
            for crop, mask in zip(crops, candidate_masks)
            if switch_model(crop) and mask.sum() > 0
        ],
        crops,
        crops_idcs,
    )


def crop_from_mask(mask: Mask, data: GrayImage) -> Tuple[GrayImage, xr.DataArray]:
    """Crop data from mask."""
    # crop to mask bbox
    non_zero_idxs = np.argwhere(mask)
    start_vert, start_hor = non_zero_idxs.min(0)
    end_vert, end_hor = non_zero_idxs.max(0)
    original_crop_idcs = [start_vert, start_hor, end_vert, end_hor]
    gray_canvas = data.copy()
    crop = gray_canvas[start_vert : end_vert + 1, start_hor : end_hor + 1].astype(float)
    # subtract mean current on contour
    mean_current = np.mean(data[mask])
    crop -= mean_current
    crop_resized = cv.resize(crop, (25, 25), interpolation=cv.INTER_CUBIC)
    return to_gray(crop_resized), original_crop_idcs


def lowest_variance_label(
    X: np.ndarray, labels: np.ndarray, unique_labels: Optional[np.ndarray] = None
) -> int:
    """Get the label with the lowest variance."""
    unique_labels = unique_labels if unique_labels is not None else np.unique(labels)
    return min(
        unique_labels, key=lambda label: np.var(X[labels == label], axis=0).mean()
    )


def overlay_masks(masks: List[xr.DataArray]) -> xr.DataArray:
    """For plotting purposes. Overlay masks on top of each other."""
    result = xr.apply_ufunc(lambda m: to_gray(m.astype(np.uint8)), masks[0])
    for mask in masks[1:]:
        result = xr.apply_ufunc(mask_image, result, mask)
    return result


def mask_at_boundary(mask: Mask) -> bool:
    """Check if mask touches the boundary of the image."""
    boundary = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]], axis=0)
    return boundary.sum() > 0
