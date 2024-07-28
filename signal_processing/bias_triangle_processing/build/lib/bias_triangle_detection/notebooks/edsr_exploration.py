from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.edsr import extract_edsr_spot, get_parallel_axis_condition, get_origin_intersection_condition, get_origin
import numpy as np
import matplotlib.pyplot as plt

path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2 (1).db"
# manual filter of relevant acquisition
ids = [110, 107, 109, 106,  104, 103]
# ids = [110]
is_edsr = lambda dataset: dataset.run_id in ids
datasets = query_datasets(path, is_edsr)
for dataset in datasets:
    plt.clf()
    data_dict = dataset.get_parameter_data()
    # keys slightly different between datasets
    key_root = "LI_" if "LI_X" in data_dict else "LI"
    x_key = key_root + "X"
    y_key = key_root + "Y"

    xarrays = dataset.to_xarray_dataset()
    data = np.sqrt(xarrays[x_key]**2 + xarrays[y_key]**2)
    data.plot()
    plt.title(f"{dataset.run_id}")
    plt.savefig(f"edsr_input_{dataset.run_id}.png")
    plt.clf()
    condition = get_parallel_axis_condition(axis=0)
    smooth, filtered, max = extract_edsr_spot(data, sigma=1, condition=condition)
    fig, ax = plt.subplots(4, 1, figsize=(5, 10))
    data.plot(ax=ax[0], yincrease=False)
    ax[0].set_title('Original')
    ax[1].imshow(smooth)
    ax[1].set_title('Adaptive gaussian thresholded')
    ax[2].imshow(filtered)
    ax[2].set_title('Filtering on connected components shapes')
    ax[3].imshow(data)
    ax[3].scatter(*max[::-1], c="r")
    ax[3].set_title('Readout point')


    plt.savefig(f"edsr_{dataset.run_id}.png")

is_edsr_check = lambda dataset: dataset.name ==  "edsr_line_scan"
datasets = query_datasets(path, is_edsr_check)
for dataset in datasets:
    plt.clf()
    data_dict = dataset.get_parameter_data()
    # keys slightly different between datasets
    key_root = "LI_" if "LI_X" in data_dict else "LI"
    x_key = key_root + "X"
    y_key = key_root + "Y"

    xarrays = dataset.to_xarray_dataset()
    data = np.sqrt(xarrays[x_key]**2 + xarrays[y_key]**2)
    origin = get_origin(data)
    condition = get_origin_intersection_condition(origin)
    smooth, filtered, max = extract_edsr_spot(data, sigma=1, relative_length=0.2, condition=condition)
    fig, ax = plt.subplots(4, 1, figsize=(5, 10))
    data.plot(ax=ax[0], yincrease=False)
    ax[0].set_title('Original')
    ax[1].imshow(smooth)
    ax[1].set_title('Adaptive gaussian thresholded')
    ax[2].imshow(filtered)
    ax[2].set_title('Filtering on connected components shapes')
    ax[3].imshow(data)
    ax[3].scatter(*max[::-1], c="r")
    ax[3].set_title('Readout point')
    plt.savefig(f"edsr_line_scan_{dataset.run_id}.png")
    