from dataclasses import dataclass
from typing import Dict, List, Tuple 

import numpy as np

from .skewed_grid import SkewedGrid

@dataclass
class MeasuredGridFeatures:
    grid: SkewedGrid
    peak_locations: np.ndarray
    x_scan: Dict[int, np.ndarray]
    y_scan: Dict[int, np.ndarray]
    x_peaks: Dict[int, List[int]]
    y_peaks: Dict[int, List[int]]
    data: np.ndarray
    measured_data: np.ndarray
    threshold: float

