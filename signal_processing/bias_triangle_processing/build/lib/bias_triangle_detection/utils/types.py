import os
from typing import Union

import numpy as np
import numpy.typing as npt

import qcodes

GrayImage = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.uint0]
Contour = npt.NDArray[np.int32]
PathLike = Union[str, bytes, os.PathLike]
QcodesDataset = qcodes.dataset.data_set.DataSet
    
