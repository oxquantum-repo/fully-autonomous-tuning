from functools import partial
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import xarray as xr
from scipy.stats import spearmanr

from bias_triangle_detection.im_utils import img_resize_gray
from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.scores import ScoreType
from bias_triangle_detection.utils.xarray_and_qcodes import map_on_dataset


def main():
    path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2.db"
    query = lambda dataset: dataset.name.startswith(f"coarse_tuning")
    all_datasets = query_datasets(path, query)
    ids = {d.name for d in all_datasets}
    for d_id in ids:
        datasets = [d for d in all_datasets if d_id in d.name]
        score_to_dataset = []
        for dataset in datasets:
            dataset = dataset.to_xarray_dataset()
            orig_shape = dataset["I_SD"].shape

            score_type = ScoreType.N_EDGES
            # There is one broken dataset in the db
            try:
                scored_array = rank(dataset, orig_shape, score_type)
            except:
                continue
            score = scored_array["score"].values[0]
            score_to_dataset.append((score, dataset))

        score_to_dataset.sort(key=lambda x: x[0])
        step = 10
        n_plots = int(np.ceil(len(score_to_dataset) / step))
        for i in range(n_plots):
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            axes = axes.flatten()
            for j, ax in enumerate(axes.flat):
                if i * step + j >= len(score_to_dataset):
                    break
                score, dataset = score_to_dataset[i * step + j]
                dataset["I_SD"].plot(ax=ax)
                ax.set_title(score)
            plt.savefig(f"{d_id}_{i}.png")
            plt.clf()


def old_main():
    experiment_name = "coarse_tunning_score"
    mlflow.set_experiment(experiment_name)

    dataset: xr.Dataset = xr.open_dataset("data/coarse_tunning_data.nc")
    # is_triangle = lambda x: True
    is_triangle = 0
    filtered_dataset = dataset.filter_by_attrs(is_triangle=is_triangle)
    is_triangle = "all" if isinstance(is_triangle, Callable) else is_triangle

    orig_size = 150

    score_type = ScoreType.N_COMPONENTS
    scored_array = rank(filtered_dataset, (orig_size, orig_size), score_type)

    plot_and_log(scored_array, is_triangle, score_type)


def rank(
    filtered_dataset: xr.Dataset, shape: Tuple[int, int], score_type: ScoreType
) -> xr.DataArray:
    func = partial(img_resize_gray, shape=shape)
    filtered_dataset = map_on_dataset(filtered_dataset, func)
    data_array = xr.concat(filtered_dataset.values(), dim="score")
    score_func = score_type.value
    scores = np.array([score_func(arr) for arr in data_array.values])
    data_array = data_array.assign_coords({"score": scores})
    return data_array


def plot_and_log(
    data_array: xr.DataArray, is_triangle: Union[str, int], score_type: ScoreType
):
    # list of rank indices that order scores
    # get indices that order scores
    data_array = data_array.sortby("score")
    plots_per_file = 10
    mlflow.log_params({"score_type": score_type.name, "is_triangle": is_triangle})
    # filtered =
    for i in range(0, len(data_array), plots_per_file):
        if len(data_array[i : i + plots_per_file]) == 1:
            data_array[i : i + plots_per_file].plot()
            continue
        data_array[i : i + plots_per_file].plot(
            x=data_array.dims[1],
            y=data_array.dims[2],
            col=data_array.dims[0],
            col_wrap=5,
        )
        filepath = f"data/plots/avg_number_of_edges_is_triangle_{is_triangle}_slice_{i}_score_{score_type.name}.png"
        plt.savefig(filepath)
        mlflow.log_artifact(filepath)

    # scores = [score(arr) for arr in dataset.values]


if __name__ == "__main__":
    main()
