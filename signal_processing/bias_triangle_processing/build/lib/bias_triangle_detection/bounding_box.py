from typing import Callable, Iterable, List, Set, Tuple
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import xarray as xr

from dataclasses import dataclass
from bias_triangle_detection.scores.separation import filter_by_current
from bias_triangle_detection.utils.contours_and_masks import get_individual_components
from bias_triangle_detection.utils.xarray_and_qcodes import xr_to_mask
from bias_triangle_detection.switches.characterisation import get_masks_as_xr
from bias_triangle_detection.bayesian_optimization.bayesian_optimizer import EvaluationResult
from bias_triangle_detection.switches.fastai_model_wrapper import FastAISwitchModel
from bias_triangle_detection.utils.types import PathLike, QcodesDataset


@dataclass
class BoundingBox:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    triangles: Set[int]
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Return the union of this bounding box with another."""
        return BoundingBox(
            x_min=min(self.x_min, other.x_min),
            x_max=max(self.x_max, other.x_max),
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
            triangles=self.triangles | other.triangles,
    )
    
    def intersects(self, mask: np.ndarray) -> bool:
        """Return True if the bounding box intersects with the mask."""
        return np.any(self.intersection(mask))
    
    def intersection(self, mask: np.ndarray) -> np.ndarray:
        """Return a mask cropped by the bounding box."""
        return mask[self.y_min : self.y_max+1, self.x_min : self.x_max+1]

    def is_good_enough(self, good_mask: np.ndarray, bad_mask: np.ndarray, area_threshold: float = 0.75) -> bool:
        good_area = np.sum(self.intersection(good_mask))
        bad_area = np.sum(self.intersection(bad_mask))
        return good_area / (good_area + bad_area) >= area_threshold

    @property
    def ordered_triangles(self) -> Tuple:
        """Return the triangles in the bounding box in a sorted tuple."""
        return tuple(sorted(self.triangles))
    
    def update_triangles(self, labels: np.ndarray, id_to_quantity: dict) -> None:
        """Update the triangles in the bounding box."""
        intersection_quantities = dict(zip(*np.unique(self.intersection(labels), return_counts=True)))
        for id, quantity in intersection_quantities.items():
            if id_to_quantity[id] == quantity:
                self.triangles.add(id)
        
def get_switches_mask(corner_pts_mask: xr.DataArray, filtered_candidates_mask: xr.DataArray) -> np.ndarray:
    """Return the mask of the switches."""
    switches_and_boundaries = corner_pts_mask - filtered_candidates_mask
    components = get_individual_components(xr_to_mask(switches_and_boundaries))[0]
    switches_mask = np.zeros_like(switches_and_boundaries)
    for c in components:
        # add only components that are not on the boundary
        if not (np.any(c[[0, -1]]) or np.any(c[:, [0, -1]])):
            switches_mask += c
    return switches_mask > 0
            
def extract_largest_bounding_boxes(corner_pts_mask: xr.DataArray, filtered_candidates_mask: xr.DataArray, readout_data: xr.DataArray, area_treshold: float = 0.75) -> Iterable[BoundingBox]:
    """ Computes the largest bounding boxes wrt to the number of triangles that do not contain any switches. """ 
    switches_mask = get_switches_mask(corner_pts_mask, filtered_candidates_mask)
    mask = xr_to_mask(filtered_candidates_mask)
    masks, labels = get_individual_components(mask)
    masks, above_current_mask = filter_by_current(readout_data, masks)
    id_to_quantity = dict(zip(*np.unique(labels, return_counts=True)))

    # compute bb per triangle
    bounding_boxes = []
    # same indexing of the id_to_quantity above
    for i, m in enumerate(masks, start=1):
        x,y,w,h = cv.boundingRect(m)
        bb = BoundingBox(x,x+w, y, y+h, {i})
        if not bb.intersects(switches_mask):
            bounding_boxes.append(bb)
    # merge bounding boxes until no more merges are possible
    # bounding_boxes stored in dictionaries to avoid duplicates by permutation of triangles
    new_bounding_boxes = {}
    good_mask = mask > 0
    while True: 
        for bb in bounding_boxes:
            for bb2 in bounding_boxes:
                if bb2.triangles.issubset(bb.triangles):
                    continue
                union = bb.union(bb2)
                if not union.is_good_enough(good_mask, switches_mask, area_treshold) or union.intersects(above_current_mask):
                    continue
                union.update_triangles(labels, id_to_quantity)
                new_bounding_boxes[union.ordered_triangles] = union
        if not new_bounding_boxes:
            break
        bounding_boxes = list(new_bounding_boxes.values())
        # filtering the bounding boxes by the number of triangles
        bounding_boxes.sort(key = lambda bb: len(bb.triangles), reverse=True)
        # arbitrary limit of 100 bounding boxes to bound complexity
        bounding_boxes = bounding_boxes[:100]
        new_bounding_boxes = {}
    
    return bounding_boxes


def scan_to_filtered_evaluation_results(
    dataset: QcodesDataset,
    full_scan: xr.DataArray,
    switch_model: FastAISwitchModel,
    scorer: 'Scorer',
    slow_axis: str = "rp",
    threshold_method: str = "default",
    min_triangles: int = 3,
    area_threshold: float = 0.75
) -> Tuple[List[EvaluationResult], plt.Figure]:
    """Obtain the filtered bounding boxes as evaluation results.

    Args:
        dataset (xr.Dataset): xarray of scan
        switch_model (FastAISwitchModel): FastAISwitchModel to use for determining switches
        slow_axis (str, optional): axis perpendicular to switches. Defaults to "rp".
        threshold_method (str, optional): outlier threshold method. Defaults to "default".
        min_triangles (int, optional): minimum number of triangles in bounding box. Defaults to 3.
        area_threshold (float, optional): minimum portion of the area covered by "good" triangles. Defaults to 0.75.

    Returns:
        Tuple[List[EvaluationResult], plt.Figure]: Scored bounding boxes as evaluation results and debug figure
    """
    bounding_boxes, filtered_candidates_mask, fig = scan_to_bounding_boxes(full_scan, switch_model, slow_axis, threshold_method, area_threshold=area_threshold)
    filtered_bb = [bb for bb in bounding_boxes if len(bb.triangles) >= min_triangles]
    evaluation_results = bounding_boxes_to_evaluations(filtered_bb, scorer, dataset, full_scan, filtered_candidates_mask)
    return evaluation_results, fig

def scan_to_bounding_boxes(
    full_scan: xr.DataArray,
    switch_model: FastAISwitchModel,
    slow_axis: str = "rp",
    threshold_method: str = "default",
    area_threshold: float = 0.75,
) -> Tuple[List[BoundingBox], xr.DataArray, plt.Figure]:
    """Obtain all the maximal bounding boxes, the candidates mask and debug figure.

    Args:
        dataset (xr.Dataset): xarray of scan
        switch_model (FastAISwitchModel): FastAISwitchModel to use for determining switches
        slow_axis (str, optional): axis perpendicular to switches. Defaults to "rp".
        threshold_method (str, optional): outlier threshold method. Defaults to "default".
        min_triangles (int, optional): minimum number of triangles in bounding box. Defaults to 3.
        area_threshold (float, optional): minimum portion of the area covered by "good" triangles. Defaults to 0.75.

    Returns:
        Tuple[List[BoundingBox], xr.DataArray, plt.Figure]: Maximal bounding boxes, the candindates mask and debug figure
    """
    (
        filtered_candidates_mask,
        _,
        corner_pts_mask,
        switch_mask,
        _,
        _,
        _,
        _
    ) = get_masks_as_xr(full_scan, slow_axis, threshold_method, switch_model)[0]
    bounding_boxes = extract_largest_bounding_boxes(corner_pts_mask, filtered_candidates_mask, full_scan, area_threshold)
    fig = plot_bounding_boxes(corner_pts_mask, filtered_candidates_mask, switch_mask, full_scan, bounding_boxes)
    return bounding_boxes, filtered_candidates_mask, fig


def plot_bounding_boxes(corner_pts_mask: xr.DataArray,
                        filtered_candidates_mask: xr.DataArray,
                        switch_mask: xr.DataArray,
                        dataset: QcodesDataset,
                        bounding_boxes: Iterable[BoundingBox]) -> plt.Figure:
    switches_mask = get_switches_mask(corner_pts_mask, filtered_candidates_mask)
    masks = get_individual_components(xr_to_mask(filtered_candidates_mask))[0]
    masks = filter_by_current(dataset, masks)[0]
    if masks:
        low_current_mask = ((np.sum(masks, axis=0)> 0)*255).astype(np.uint8)
    else:
        low_current_mask = np.zeros_like(filtered_candidates_mask).astype(np.uint8)
    fig, axes = plt.subplots(1, 5, figsize=(13, 5))
    axes[0].imshow(filtered_candidates_mask)
    axes[0].set_title("No switches masks")
    axes[1].imshow(switches_mask)
    axes[1].set_title("switches (current NN)")
    for bb in bounding_boxes:
        cv.rectangle(low_current_mask,(bb.x_min,bb.y_min),(bb.x_max,bb.y_max),100,1)
    axes[2].imshow(low_current_mask)
    axes[2].set_title("Low current + bbs")
    axes[3].imshow(switch_mask.to_numpy())
    axes[3].set_title("switches (old analytical)")
    axes[4].imshow(dataset.to_numpy())
    axes[4].set_title("Measurement")
    return fig 


def bounding_boxes_to_evaluations(bounding_boxes: Iterable[BoundingBox], scorer: Callable, dataset: QcodesDataset, full_scan: xr.DataArray, mask: xr.DataArray) -> List[EvaluationResult]:
    """Compute score per bounding box."""
    evaluation_results = []
    for bb in bounding_boxes:
        cropped_scan = full_scan[bb.y_min:bb.y_max+1, bb.x_min:bb.x_max+1]
        cropped_mask = mask[bb.y_min:bb.y_max+1, bb.x_min:bb.x_max+1]
        evaluation_results.append(scorer(dataset=dataset, cropped_scan=cropped_scan, scorer_kwargs={"triangles_mask": cropped_mask}))
    return evaluation_results
