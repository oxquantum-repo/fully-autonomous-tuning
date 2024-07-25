import matplotlib.pyplot as plt
import xarray as xr

from bias_triangle_detection.switches.characterisation import (
    create_viz_plot,
    get_masks_as_xr,
)
from bias_triangle_detection.utils.numpy_to_image import to_gray


def switch_characterization_plot(
    dataset: xr.Dataset,
    slow_axis: str = "rp",
    threshold_method: str = "default",
    **switches_kwargs,
) -> plt.Figure:
    """Obtain list of directions for candidates in plunge space.

    Args:
        dataset (xr.Dataset): xarray of scan
        slow_axis (str, optional): axis perpendicular to switches. Defaults to "rp".
        threshold_method (str, optional): outlier threshold method. Defaults to "default".

    Returns:
        Tuple[List[Tuple[Direction, Direction]], plt.Figure]: List of directions in plunge space and debug figure
    """
    masks, axes_dims = get_masks_as_xr(
        dataset, slow_axis, threshold_method, **switches_kwargs
    )
    (
        filtered_candidates_mask,
        candidate_mask,
        corner_pts_mask,
        switch_mask,
        outlier_mask,
        gray_diff,
        _,
        _
    ) = masks
    slow_axis, fast_axis = axes_dims
    fig = create_viz_plot(
        xr.apply_ufunc(to_gray, dataset),
        filtered_candidates_mask,
        candidate_mask,
        corner_pts_mask,
        switch_mask,
        outlier_mask,
        gray_diff,
        slow_axis,
        fast_axis,
    )
    return fig


if __name__ == "__main__":
    import datetime

    import matplotlib.pyplot as plt
    import mlflow
    import xarray as xr

    from bias_triangle_detection.qcodes_db_utils import query_datasets
    from bias_triangle_detection.switches.fastai_model_wrapper import (
        ArrayGetter,
        FastAISwitchModel,
        LabelGetter,
    )

    path = "data/GeSiNW_Qubit_VTI01_Jonas_3.db"
    time_ago = {"hours": 2, "days": 6}
    time_thresh = (datetime.datetime.now() - datetime.timedelta(**time_ago)).timestamp()
    query = (
        lambda d: d.name.startswith("coarse_tuning")
        and (d.run_timestamp_raw > time_thresh)
        and (d.to_xarray_dataset()["I_SD"].shape == (100, 100))
    )

    # datasets = query_datasets(path, lambda ds: ds.guid in guids)
    datasets = query_datasets(path, query)
    # datasets = query_datasets(path, lambda ds: ds.name.endswith(task_id_high_res))
    mlflow.set_experiment("switches")

    # get threshold finder
    switches_kwargs = {
        "sigma": 0.6,
        "min_switch_size": 4,
        "max_switch_size": 10,
    }
    THRESHOLD_METHOD = "default"

    mlflow.set_tag("threshold_method", THRESHOLD_METHOD)
    mlflow.log_params({**switches_kwargs})

    path_learner = "models/export.pkl"
    path_checkpoint = "best_model_2"
    switch_model = FastAISwitchModel(path_learner, path_checkpoint)
    switches_kwargs["switch_model"] = switch_model

    # get switches
    for qcodes_dataset in datasets:
        dataset = qcodes_dataset.to_xarray_dataset()["I_SD"]
        rp, lp = list(dataset.dims)

        fig = switch_characterization_plot(
            dataset,
            threshold_method=THRESHOLD_METHOD,
            **switches_kwargs,
        )

        mlflow.log_figure(fig, f"switches_{qcodes_dataset.guid}.png")
