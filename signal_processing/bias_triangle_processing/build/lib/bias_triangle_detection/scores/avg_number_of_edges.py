from enum import Enum
from functools import partial
from typing import Callable, List

import cv2 as cv
import numpy as np

from bias_triangle_detection.scores.base import ScoreResult, extract_triangles


def score_number_of_components(
    readout_data: np.ndarray,
) -> ScoreResult:
    masks = simple_thresh(readout_data)
    ret, _ = cv.connectedComponents(masks)
    return ret - 1

def score_number_of_edges(
    readout_data: np.ndarray,
    *,
    res_h: int = 4,
    min_area_f: float = 0.001,
    thr_method: str = "triangle",
    denoising: bool = True,
    allow_MET: bool = False,
    direction: str = "down",
    score_statistic: Callable[[List[float]], ScoreResult] = np.mean,
) -> ScoreResult:
    _, triangles_masks = extract_triangles(
        readout_data,
        res=res_h,
        min_area_f=min_area_f,
        thr_method=thr_method,
        denoising=denoising,
        allow_MET=allow_MET,
        direction=direction,
        inv=True,
    )
    score_per_component = [count_edges(t) for t in triangles_masks]
    return score_statistic(score_per_component)


def count_edges(mask: np.ndarray, eps: float = 0.01) -> int:
    c, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    peri = cv.arcLength(c[0], True)
    approx = cv.approxPolyDP(c[0], eps * peri, True)
    no_edges = len(approx)

    return no_edges


def simple_thresh(gray: np.ndarray) -> np.ndarray:
    # assume gray is normalized
    _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)

    return threshold
