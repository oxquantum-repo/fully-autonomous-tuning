from typing import List, Protocol, Tuple, TypeVar

import cv2 as cv
import numpy as np

from bias_triangle_detection.btriangle_detection import triangle_segmentation_alg
from bias_triangle_detection.utils.contours_and_masks import get_individual_components

ScoreResult = TypeVar("ScoreResult")

def extract_triangles(
    readout_data: np.ndarray,
    res_h: int = 4,
    min_area_f: float = 0.001,
    thr_method: str = "triangle",
    denoising: bool = True,
    allow_MET: bool = False,
    direction: str = "down",
    inv: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
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
    return gray_data, individual_components


class ScoreFunction(Protocol):
    def __call__(self, readout_data: np.ndarray, **kwargs) -> ScoreResult:
        pass
