from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Number
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.stats import entropy

from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.scores.base import ScoreResult

OptionalQuantity = Union[Number, np.ndarray, Sequence]


class MagnitudeType(Enum):
    PCA = 0
    EUCLIDEAN = 1


def PCA(*arrays):
    arrays_copy = [np.copy(array) for array in arrays]
    for array in arrays_copy:
        array[np.isnan(array)] = np.mean(np.ma.masked_invalid(array))

    Z = np.stack(arrays_copy, axis=-1)
    shape = Z.shape

    # summing over every axis except the last
    u = np.mean(Z, axis=tuple(range(0, shape.__len__() - 1)), keepdims=True)

    B = (Z - u).reshape(np.product(shape[0:-1]), shape[-1])
    C = np.einsum("ki, kj -> ij", B, B)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    arg_sorted = np.flip(eigen_values.argsort())
    eigen_vectors = eigen_vectors[:, arg_sorted]
    pca = np.einsum("ik, kj -> ij", B, eigen_vectors).reshape(shape)[..., 0]
    return pca


def _magnitude(
    readout_data: np.ndarray,
    rescale: bool = True,
    with_clip: bool = True,
    magnitude_type: MagnitudeType = MagnitudeType.EUCLIDEAN,
) -> np.ndarray:
    if magnitude_type == MagnitudeType.EUCLIDEAN:
        norm = np.linalg.norm(readout_data, axis=-1)
    elif magnitude_type == MagnitudeType.PCA:
        norm = PCA(*np.swapaxes(readout_data, 0, 1))
    if with_clip:
        norm = np.clip(norm, np.median(norm), None)
    min_value = norm.min(-1)
    _range = norm.max(-1) - min_value
    if rescale:
        norm = (norm - min_value[..., None]) / _range[..., None]
        return norm, 1
    return norm, _range


@dataclass
class EntropyScore:
    """scipy entropy score with rescaling"""

    rescale: bool = True
    with_clip: bool = True
    magnitude_type: Literal["PCA", "EUCLIDEAN"] = "EUCLIDEAN"

    def __call__(self, readout_data: np.ndarray) -> ScoreResult:
        norm, _ = _magnitude(
            readout_data,
            self.rescale,
            self.with_clip,
            MagnitudeType[self.magnitude_type],
        )
        return entropy(norm)


@dataclass
class prominence_score:
    """Prominence score with default value if no peaks are found.
    supports rescaling."""

    height: OptionalQuantity = None
    threshold: OptionalQuantity = None
    distance: Optional[Number] = None
    prominence: float = 0.5
    width: OptionalQuantity = None
    wlen: Optional[int] = None
    rel_height: Optional[float] = None
    plateau_size: OptionalQuantity = None
    _default_score: float = 0.0

    def __call__(self, readout_data: np.ndarray) -> ScoreResult:
        _, prominence = self._call(readout_data)
        return prominence

    def _call(self, readout_data: np.ndarray) -> ScoreResult:
        peaks, descriptions = self._find_peaks(readout_data)
        peak_locs = np.zeros(len(readout_data))
        peak_locs[peaks] = 1
        if len(descriptions["prominences"]) == 0:
            return peak_locs, self._default_score
        return peak_locs, descriptions["prominences"].max()

    def _find_peaks(self, readout_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        norm, _range = _magnitude(readout_data, True)
        return find_peaks(norm, **self._find_peaks_kwargs())

    def _find_peaks_kwargs(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


if __name__ == "__main__":
    path = "data/GeSiNW_Qubit_VTI01_Jonas_2.db"
    from dataclasses import asdict

    import mlflow

    from bias_triangle_detection.qcodes_db_utils import query_datasets

    # get datasets from readout optim
    task_id = "84cb524d-2916-43ce-a275-4e1c0baadfdc"
    datasets_readout_optim = query_datasets(path, lambda ds: task_id in ds.name)

    last_task_id = datasets_readout_optim[-1].name.split("_")[-1]

    datasets_last_run = query_datasets(path, lambda ds: last_task_id in ds.name)

    prom_scorer = prominence_score()
    scorer = EntropyScore()

    # concatenate them along new dimension
    data_array = xr.concat(
        map(lambda ds: ds.to_xarray_dataset(), datasets_last_run), dim="rank"
    )

    # compute norm
    data_array["norm"] = xr.apply_ufunc(
        lambda x: _norm(x)[0],
        data_array.to_array(),
        input_core_dims=[["variable"]],
        exclude_dims={"variable"},
    )
    mf, rank = list(data_array.dims)

    # compute entropy and assign as coordinate
    res = xr.apply_ufunc(
        entropy,
        data_array["norm"],
        input_core_dims=[[mf]],
        vectorize=True,
    )
    data_array = data_array.assign_coords({rank: res})
    data_array = data_array.sortby(rank, ascending=True)

    # compute peaks
    locs, prominences = xr.apply_ufunc(
        prom_scorer._call,
        data_array[["LIX", "LIY"]].to_array(),
        input_core_dims=[[mf, "variable"]],
        output_core_dims=[[mf], []],
        vectorize=True,
    )
    data_array["locs"] = locs
    data_array = data_array.assign_coords({"prominence": ("rank", prominences.data)})
    # filtered = xr.where(data_array["locs"], data_array["norm"], 0)
    # # plots = filtered[:10].plot(x=mf, col=rank, col_wrap=5)
    plots_per_file = 10
    mlflow.set_experiment("peaks_and_entropy")
    mlflow.log_params(asdict(prom_scorer))
    for i in range(0, data_array[rank].shape[0], plots_per_file):
        plots = data_array["norm"][i : i + plots_per_file].plot(
            x=mf,
            col=rank,
            col_wrap=5,
        )
        for ax, name_dict in zip(plots.axes.flat, plots.name_dicts.flat):
            data = data_array.loc[name_dict]
            mf_vals = data.where(data["locs"], drop=True)[mf]
            for mf_val in mf_vals:
                ax.vlines(mf_val, 0, 1, color="r")
        mlflow.log_figure(plots.fig, f"peaks_and_entropy_{i}.png")
