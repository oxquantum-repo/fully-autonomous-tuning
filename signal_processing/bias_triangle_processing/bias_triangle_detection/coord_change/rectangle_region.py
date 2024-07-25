# images to segment
from typing import Optional, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qcodes.dataset import DataSetProtocol

from bias_triangle_detection.readout_search_rectangle import (
    RotatedRectangle,
    get_rectangle_features,
)
from bias_triangle_detection.utils.contours_and_masks import mask_image, to_gray
from bias_triangle_detection.utils.xarray_and_qcodes import (
    add_detection_dataset_scatter,
    pixel_point_to_coord_space,
    xarrays_to_visualize,
)


def get_min_area(
    shape: Tuple[int, int],
    res: int,
    min_area: Optional[float] = None,
    relative_min_area: Optional[float] = None,
):
    if min_area is not None:
        return min_area
    if relative_min_area is not None:
        return res**2 * shape[0] * shape[1] * relative_min_area
    raise ValueError("Either min_area or relative_min_area must be specified")


def prepare_args(
    shape: Tuple[int, int],
    res: int,
    bias_direction: str,
    min_area: float = None,
    relative_min_area: float = None,
    invert: bool = True,
    compensated_gates_list: list = None,
):
    min_area = get_min_area(
        shape,
        res,
        min_area=min_area,
        relative_min_area=relative_min_area,
    )
    invert = -1 if invert else 1
    prior_dir = None
    if 'V_RP' in compensated_gates_list:
        if bias_direction == "positive_bias":
            prior_dir = "up"
        if bias_direction == "negative_bias":
            prior_dir = "right"
        if prior_dir is None:
            raise ValueError("bias_direction must be either positive_bias or negative_bias")
    else:
        if bias_direction == "positive_bias":
            prior_dir = "right"
        if bias_direction == "negative_bias":
            prior_dir = "up"
        if prior_dir is None:
            raise ValueError("bias_direction must be either positive_bias or negative_bias")
    kwargs = {
        "res": res,
        "min_area": min_area,
        "direction": "down",
        "invert": invert,
        "prior_dir": prior_dir,
    }
    return kwargs


def images_to_rectangle(
    unpulsed_ds: DataSetProtocol,
    pulsed_ds: DataSetProtocol,
    res: int,
    bias_direction: str,
    min_area: float = None,
    relative_min_area: float = None,
    thr_method: str = "triangle",
    invert: bool = True,
    create_rectangle_ds: bool = True,
    create_viz_plot: bool = True,
    compensated_gates_list: list = None
) -> Tuple[RotatedRectangle, Optional[DataSetProtocol], Optional[plt.Figure]]:
    # current
    readout = unpulsed_ds.dependent_parameters[0]
    data_var = readout.name
    unpulsed = unpulsed_ds.to_xarray_dataset()[data_var]
    pulsed = pulsed_ds.to_xarray_dataset()[data_var]
    # right plunger, left plunger. Transpose according to bias direction
    rp, lp = sorted(pulsed.dims, key=lambda x: "LP" in x)
    first_dim, second_dim = rp, lp
    if bias_direction == "negative_bias":
        first_dim, second_dim = lp, rp
    unpulsed = unpulsed.transpose(first_dim, second_dim)
    pulsed = pulsed.transpose(first_dim, second_dim)

    # kwargs for get_rectangle_features
    kwargs = prepare_args(
        unpulsed.shape,
        res,
        bias_direction,
        min_area,
        relative_min_area,
        invert,
        compensated_gates_list
    )
    kwargs["unpulsed"] = unpulsed.to_numpy()
    # kwargs['prior_dir'] = prior_dir
    print(f'using kwargs: {kwargs}')

    (
        clipped_excited_line,
        point_base,
        clipped_excited_line_mask,
        point_base_mask,
    ) = xr.apply_ufunc(
        get_rectangle_features,
        pulsed,
        input_core_dims=[[first_dim, second_dim]],
        output_core_dims=[
            ["excited_line", "coords"],
            ["base_line", "coords"],
            [first_dim, second_dim],
            [first_dim, second_dim],
        ],
        kwargs=kwargs,
    )
    fig = None
    if create_viz_plot:
        gray = xr.apply_ufunc(to_gray, pulsed, keep_attrs=True)
        viz = {
            "gray": gray,
            "line_excited": xr.apply_ufunc(mask_image, gray, clipped_excited_line_mask),
            "base_line": xr.apply_ufunc(mask_image, gray, point_base_mask),
        }
        viz = xarrays_to_visualize(viz)
        plots = viz.plot(x=rp, y=lp, col="viz")
        fig = plots.fig
    points = np.concatenate(
        [point_base.to_numpy().reshape(-1, 2), clipped_excited_line.to_numpy()],
        axis=0,
    )
    if bias_direction == 'negative_bias':
        points = points[:, ::-1]
    points_volt = pixel_point_to_coord_space(points, pulsed.coords)
    rectangle = rectangle_in_volt_space(points_volt)

    if not create_rectangle_ds:
        return rectangle, None, fig
    box_points = cv.boxPoints(rectangle)
    detection_ds = add_detection_dataset_scatter(
        pulsed_ds, box_points, name="rectangle_detection"
    )
    add_metadata_to_rectangle(
        detection_ds,
        rectangle,
        pulsed_ds.guid,
        unpulsed_ds.guid,
        res=res,
        bias_direction=bias_direction,
        min_area=min_area,
        relative_min_area=relative_min_area,
        thr_method=thr_method,
        invert=invert,
    )
    return rectangle, detection_ds, fig


def rectangle_in_volt_space(
    points: np.ndarray, resolution: float = 1e-4
) -> RotatedRectangle:
    (x, y), (w, h), angle = cv.minAreaRect((points / resolution).round().astype(int))
    return (x * resolution, y * resolution), (w * resolution, h * resolution), angle


def add_metadata_to_rectangle(
    detection_ds: DataSetProtocol,
    rectangle: RotatedRectangle,
    pulsed_guid: str,
    unpulsed_guid: str,
    **kwargs,
):
    (x, y), (w, h), angle = rectangle
    names = [
        "rectangle_x",
        "rectangle_y",
        "rectangle_w",
        "rectangle_h",
        "rectangle_angle",
    ]
    for name, value in zip(names, [x, y, w, h, angle]):
        detection_ds.add_metadata(name, value)
    detection_ds.add_metadata("pulsed_guid", pulsed_guid)
    detection_ds.add_metadata("unpulsed_guid", unpulsed_guid)
    for name, value in kwargs.items():
        if value is None:
            value = ""
        detection_ds.add_metadata(name, value)


if __name__ == "__main__":
    import pickle

    import cv2 as cv
    import matplotlib.pyplot as plt
    from qcodes.dataset import (
        DataSetProtocol,
        initialise_or_create_database_at,
        load_by_guid,
        load_by_id,
        plot_dataset,
    )

    db_path = "data/GeSiNW_Qubit_VTI01_Jonas_2.db"
    initialise_or_create_database_at(db_path)
    ds_id = 1939
    dataset = load_by_id(ds_id)
    # plot_dataset(dataset)
    pulsed = load_by_guid(dataset.metadata["pulsed_guid"])
    unpulsed = load_by_guid(dataset.metadata["unpulsed_guid"])
    res = dataset.metadata["res"]
    min_area = dataset.metadata["min_area"]
    bias_direction = dataset.metadata["bias_direction"]
    ((x, y), (w, h), angle), detection_ds, fig = images_to_rectangle(
        unpulsed,
        pulsed,
        res,
        bias_direction,
        min_area,
    )
    fig, ax = plt.subplots(1)
    axs, cbs = plot_dataset(pulsed, axes=ax)
    axs, cbs = plot_dataset(detection_ds, axes=ax, c="black")
    plt.show()
    print(1)
