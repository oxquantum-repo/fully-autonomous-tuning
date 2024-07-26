from typing import Callable

import mlflow
import numpy as np
import xarray as xr
from scipy.stats import spearmanr
from score_on_coulomb_peaks import plot_and_log, rank

from bias_triangle_detection.coarse_tunning.scores import ScoreType

EPSILON = 1e-6


def main():
    experiment_name = "rank_vs_shape"
    mlflow.set_experiment(experiment_name)

    dataset: xr.Dataset = xr.open_dataset("data/coarse_tunning_data.nc")
    # is_triangle = lambda x: True
    is_triangle = 1
    filtered_dataset = dataset.filter_by_attrs(is_triangle=is_triangle)
    is_triangle = "all" if isinstance(is_triangle, Callable) else is_triangle

    orig_scale_up = 4
    orig_size = 150 * orig_scale_up
    n_sizes = 20
    sizes = np.linspace(0.2, 1, 20)
    ranks = np.zeros((n_sizes + 1, len(filtered_dataset)))

    score_type = ScoreType.N_EDGES
    ranks[0] = rank(filtered_dataset, (orig_size, orig_size), score_type)[
        "score"
    ].values

    for i, _size in enumerate(sizes):
        shape = (int(orig_size * _size), int(orig_size * _size))
        data_array = rank(filtered_dataset, shape, score_type)
        ranks[i + 1] = data_array["score"].values
    spearmans = np.zeros((n_sizes,))
    for i, _size in enumerate(sizes):
        spearmans[i] = spearmanr(ranks[0], ranks[i + 1]).correlation
        mlflow.log_metric("spearman", spearmans[i], step=i)
        mlflow.log_metric("size", _size, step=i)
    mlflow.log_param("original_scale_up", orig_scale_up)

    # plot_and_log(data_array, is_triangle, ScoreType.N_EDGES)


if __name__ == "__main__":
    main()
