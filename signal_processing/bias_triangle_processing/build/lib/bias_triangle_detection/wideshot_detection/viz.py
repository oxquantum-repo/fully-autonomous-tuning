from typing import Optional

import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np 
from copy import copy

# from mlflow_utils.optional_mlflow import MLFlowLogger as mlflow

def plot_measurement_mask(mask: np.ndarray, fname: str, title: str="") -> None:
    fig, ax = plt.subplots(figsize=(15,15))
    pos = ax.imshow(mask, interpolation = 'none')
    fig.colorbar(pos, ax=ax)
    ax.set_title(title)
    fig.savefig(fname)
    # mlflow.log_artifact(fname)
    plt.close(fig)

def plot_measurement_histogram(data: np.ndarray, fname: str, title: str="", threshold: Optional[float]=None) -> None:
    fig, ax = plt.subplots(figsize=(15,15))
    bins = max(10, int(np.sqrt(data.size))) 
    ax.hist(data.flatten(), bins=bins, density=True)
    if threshold is not None:
        ax.axvline(threshold, c='r', ls='--')
    fig.savefig(fname)
    plt.close(fig)

def plot_detected_contor_locations(data: np.ndarray, contour_locations: np.ndarray, estimated_contour_locations: np.ndarray, fname:str, title: str="") -> None:    
    fig, ax = plt.subplots(figsize=(15,15))
    pos = ax.imshow(data, interpolation = 'none')
    fig.colorbar(pos, ax=ax)
    if len(contour_locations) > 0:
        ax.scatter(contour_locations[:, 1], contour_locations[:, 0], c='b',marker='o')
    if estimated_contour_locations is not None and len(estimated_contour_locations) > 0:
        ax.scatter(estimated_contour_locations[:, 1], estimated_contour_locations[:, 0], c='r',marker='x')
    ax.set_title(title)
    fig.savefig(fname)
    plt.close(fig)


def create_debug_plots_for_wideshot_sampler(wideshot_sampler: "WideshotSampler",  data: np.ndarray, file_prefix: str) -> None:
    measured_data = np.zeros_like(data)
    measured_data[wideshot_sampler.measurement_mask] = data[wideshot_sampler.measurement_mask]
    measured_data[~wideshot_sampler.measurement_mask] = np.nan 
    plot_measurement_mask(measured_data, f"{file_prefix}_measured_locations.png")    
    plot_detected_contor_locations(
        measured_data, 
        wideshot_sampler.contour_locations, 
        np.array(wideshot_sampler.grid_points),
        fname=f"{file_prefix}_contour_locations_on_measurements.png"
    )

    data_above_thresh = copy(data)
    data_above_thresh[data_above_thresh < wideshot_sampler.threshold] = np.nan
    plot_measurement_mask(data_above_thresh, f"{file_prefix}_data_above_threshold.png")
    plot_detected_contor_locations(
        data_above_thresh, 
        wideshot_sampler.contour_locations, 
        np.array(wideshot_sampler.grid_points),
        fname=f"{file_prefix}_contour_locations_on_data_above_threshold.png"
    )
    plot_measurement_mask(wideshot_sampler.measurement_mask, f"{file_prefix}_measurement_mask_final.png")
    plot_measurement_histogram(
        data[data > wideshot_sampler.threshold], 
        f"{file_prefix}_histogram_measured_values_above_threshold.png",
        threshold=wideshot_sampler.threshold
    )
    plot_measurement_histogram(
        data[wideshot_sampler.measurement_mask], 
        f"{file_prefix}_histogram_measured_values.png",
        threshold=wideshot_sampler.threshold
    )
    plot_measurement_histogram(
        data, 
        f"{file_prefix}_histogram_all_data.png",
        threshold=wideshot_sampler.threshold
    )
    plot_measurement_mask(wideshot_sampler.binary_data, f"{file_prefix}_binary_data_final.png")
    plot_measurement_mask(data >= wideshot_sampler.threshold, f"{file_prefix}_binary_data_all.png")