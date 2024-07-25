from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
import xarray as xr
from qcodes.dataset import DataSetProtocol, Measurement
from xarray.core.coordinates import DataArrayCoordinates

Xarr = Union[xr.DataArray, xr.Dataset]


def vector_func(
    xarray: xr.DataArray,
    func: Callable[[np.ndarray], np.ndarray],
) -> xr.DataArray:
    """drop coords and apply func"""
    xarray = xarray.drop(xarray.coords)
    return xr.apply_ufunc(
        func,
        xarray,
        keep_attrs=True,
    )


def map_on_dataset(
    dataset: xr.Dataset, func: Callable[[np.ndarray], np.ndarray]
) -> xr.Dataset:
    new_dataset = {}
    for arr_name in dataset:
        new_dataset[arr_name] = vector_func(dataset[arr_name], func)
    new_dataset = xr.Dataset(new_dataset)
    new_dataset.attrs = dataset.attrs
    return new_dataset


def pixel_points_to_ds(
    points: np.ndarray, res: int, data: DataSetProtocol
) -> DataSetProtocol:
    # collection of pixel points to dataset
    dataxarr = data.to_xarray_dataset()
    readout = data.dependent_parameters[0]
    data_var = readout.name
    contour_pts = points.reshape(-1, 2) / res
    contour_pts_volt = pixel_point_to_coord_space(
        contour_pts, dataxarr[data_var].coords
    )
    return add_detection_dataset_scatter(data, contour_pts_volt)


def add_detection_dataset_scatter(
    parent_dataset: DataSetProtocol,
    points,
    name: str = "detection",
) -> DataSetProtocol:
    # collection of points to dataset
    meas = Measurement(name=name)
    domain_specs = [
        spec
        for spec in parent_dataset.paramspecs.values()
        if len(spec.depends_on_) == 0
    ]
    for spec in domain_specs:
        setpoints = (
            None
            if spec != domain_specs[-1]
            else [_spec.name for _spec in domain_specs[:-1]]
        )
        meas.register_custom_parameter(
            spec.name,
            label=spec.label,
            unit=spec.unit,
            setpoints=setpoints,
            paramtype="array",
        )
    meas.register_parent(parent=parent_dataset, link_type="rect detection")

    with meas.run() as datasaver:
        result = ((spec.name, coord) for spec, coord in zip(domain_specs, points.T))
        datasaver.add_result(*result)

    return datasaver.dataset


def pixel_point_to_coord_space(
    points: np.ndarray, coords: DataArrayCoordinates
) -> np.ndarray:
    # pixel points to volt space
    # it assumes axes have been swaped as with most triangle detection functions
    return np.array(
        [
            [
                coords[coord_name].values[int(point_index)]
                for coord_name, point_index in zip(coords, point[::-1])
            ]
            for point in points
        ]
    )


def xarrays_to_visualize(
    arrays_to_viz: Dict[str, Xarr], dimension_name: str = "viz"
) -> Xarr:
    """Create dimension viz from xarrays for visualization."""
    return xr.concat(
        arrays_to_viz.values(),
        dim=pd.Index(
            arrays_to_viz,
            name=dimension_name,
        ),
    )


def xr_to_mask(xr_data: xr.Dataset) -> np.ndarray:
    array = xr_data.to_numpy()
    assert set(array.flatten()).issubset({0, 1, 255, False, True})
    return (array > 0).astype(np.uint8)*255