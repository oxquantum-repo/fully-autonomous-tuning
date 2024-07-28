from tqdm import tqdm

import pickle
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from functools import partial
import os
from qcodes.dataset import (
    DataSetProtocol,
    experiments,
    initialise_or_create_database_at,
    load_by_id,
)

from bias_triangle_detection.btriangle_location_detection import get_locations
from bias_triangle_detection.qcodes_db_utils import check_condition, get_datasets_from_guids


def has_name(dataset: DataSetProtocol, *, name: str) -> bool:
    return name in dataset.name


def get_experiment_context(database_paths: List[str]) \
    -> List[Tuple[DataSetProtocol, DataSetProtocol, np.ndarray, np.ndarray, str]]:
    """
    Extract a list of tuples with the required experiment data from databases at the paths provided.
    :param database_paths: list of paths to data based where the data can be found.
    :return: (low_magnet_dataset, high_magnet_dataset, numpy_array_low_magnet, numpy_array_high_magnet, bias_direction)
    """
    collected = []
    number_of_fails = 0
    for path in database_paths:
        print(f'Loading database from {path}.')
        initialise_or_create_database_at(path)

        list_of_experiments = experiments()
        # there is only one exp
        experiment = list_of_experiments[0]
        datasets = experiment.data_sets()

        guids = check_condition(
            datasets, partial(has_name, name="wide_shot")
        )
        datasets = get_datasets_from_guids(guids)

        for dataset in tqdm(datasets):
            try:
                current_name = dataset.dependent_parameters[0].name
                name = dataset.name
                numpy_array = dataset.get_parameter_data(current_name)[current_name][
                    current_name
                ]
                index = dataset.run_id
                if len(numpy_array) == 0:
                    continue
                if dataset.guid in ['505bc877-0000-0000-0000-018930e041a1',
                                    '8cbc79db-0000-0000-0000-018930e54266',
                                    'c6864676-0000-0000-0000-01893106f904',
                                    'c484848b-0000-0000-0000-018992f5534a',
                                    '1803e2ee-0000-0000-0000-018950649abb']:
                    continue
                if name == 'wide_shot_low_magnet':
                    high_magnet_ds = load_by_id(index-1)
                    try:
                        numpy_high_magnet = high_magnet_ds.get_parameter_data(current_name)[current_name][
                            current_name
                        ]
                    except:
                        continue
                    bias_v_low_magnet = dataset.snapshot['station']['instruments']['LNHR_dac']['submodules']['ch8']['parameters']['volt']['value']
                    bias_v_high_magnet = high_magnet_ds.snapshot['station']['instruments']['LNHR_dac']['submodules']['ch8']['parameters']['volt']['value']

                    if numpy_array.shape == numpy_high_magnet.shape and bias_v_high_magnet == bias_v_low_magnet:
                        if dataset.snapshot['station']['instruments']['LNHR_dac']['submodules']['ch8']['parameters']['volt']['value']>0:
                            bias_direction = 'positive_bias'
                        else:
                            bias_direction = 'negative_bias'
                        collected.append((dataset, high_magnet_ds, numpy_array, numpy_high_magnet, bias_direction))
            except:
                number_of_fails += 1

    if number_of_fails > 0:
        print(f'Warning: {number_of_fails} data sets were not able to be processed.')

    return collected


def get_cutout(img: np.ndarray, blob: np.ndarray, sidelength: int = 50) -> np.ndarray:
    """
    Cuts out a square region from the image centered around the blob.

    Args:
        img (np.ndarray): The input image.
        blob (np.ndarray): The blob around which to cut out.
        sidelength (int, optional): The sidelength of the square to cut out. Defaults to 50.

    Returns:
        np.ndarray: The cut out image.
    """
    x_bottom = (
        int(blob[0] - sidelength / 2) if int(blob[0] - sidelength / 2) >= 0 else 0
    )
    x_top = int(blob[0] + sidelength / 2) if int(blob[0] + sidelength / 2) >= 0 else 0
    y_bottom = (
        int(blob[1] - sidelength / 2) if int(blob[1] - sidelength / 2) >= 0 else 0
    )
    y_top = int(blob[1] + sidelength / 2) if int(blob[1] + sidelength / 2) >= 0 else 0
    img_save = img.copy()
    img = img[x_bottom:x_top, y_bottom:y_top]
    while img.shape[0] != img.shape[1]:
        sidelength -= 1
        img = get_cutout(img_save, blob, sidelength)
    return img


def cutout_from_large_scan(img_block, img_leak, locations, sidelength=20, flip_bias = False, plot=False):
    cutouts = []
    if plot:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(img_block, origin='lower')
        axs[1].imshow(img_leak, origin='lower')
        axs[0].axis('off')
        axs[1].axis('off')
        plt.show()
    for location in locations:
        img_block_cutout = get_cutout(img_block, location, sidelength= sidelength)
        img_leak_cutout = get_cutout(img_leak, location, sidelength= sidelength)
        if flip_bias:
            img_block_cutout = img_block_cutout.T
            img_leak_cutout = img_leak_cutout.T
        if img_block_cutout.shape[0] == sidelength and img_block_cutout.shape[1] == sidelength:
            cutouts.append([img_block_cutout, img_leak_cutout])
            if plot:
                fig, axs = plt.subplots(1,2)
                axs[0].imshow(img_block_cutout, origin='lower')
                axs[1].imshow(img_leak_cutout, origin='lower')
                plt.suptitle('x: {}, y: {}'.format(location[0], location[1]))
                axs[0].axis('off')
                axs[1].axis('off')
                plt.show()
        else:
            cutouts.append(None)
    return cutouts


def get_cutouts_from_collected_data(collected):
    all_data = {}

    for i, (data_high_magnet, data_low_magnet, _, _, bias_direction) in tqdm(enumerate(collected)):
        data_high_magnet_xarray = data_high_magnet.to_xarray_dataset()
        data_low_magnet_xarray = data_low_magnet.to_xarray_dataset()

        if bias_direction == 'positive_bias':
            data_high_magnet_analysis = -data_high_magnet_xarray['I_SD'].to_numpy()
            data_low_magnet_analysis = -data_low_magnet_xarray['I_SD'].to_numpy()
        else:
            data_high_magnet_analysis = data_high_magnet_xarray['I_SD'].to_numpy()
            data_low_magnet_analysis = data_low_magnet_xarray['I_SD'].to_numpy()

        axes_values = []
        axes_values_names = []
        axes_units = []

        for item, n in dict(data_high_magnet_xarray.dims).items():
            axes_values.append(data_high_magnet_xarray[item].to_numpy())
            axes_values_names.append(data_high_magnet_xarray[item].long_name)
            axes_units.append(data_high_magnet_xarray[item].unit)

        (
            anchor,
            peaks_px,
            peaks,
            all_triangles_px,
            all_triangles,
            _,
        ) = get_locations(
            data_high_magnet_analysis,
            axes_values[1],
            axes_values[0],
            axes_values_names[1],
            axes_values_names[0],
            return_figure=True,
            plot=False,
            offset_px=13,
        )

        flip_bias = True if bias_direction == 'negative_bias' else False
        cutouts = cutout_from_large_scan(data_low_magnet_analysis,
                                         data_high_magnet_analysis,
                                         all_triangles_px,
                                         sidelength=np.max(peaks_px),
                                         flip_bias = flip_bias)
        all_data[str({'high_magnet': data_high_magnet.guid, 'low_magnet': data_low_magnet.guid})]\
            = {
            'cutouts': cutouts,
            'bias_direction': bias_direction,
            'numpy_arrays': {'high_magnet': data_high_magnet_analysis,
                             'low_magnet': data_low_magnet_analysis},
            'guids': {'high_magnet': data_high_magnet.guid, 'low_magnet': data_low_magnet.guid},
            'peaks': peaks,
            'peaks_px': peaks_px,
            'all_triangles_px': all_triangles_px,
            'all_triangles': all_triangles,
            'anchor': anchor
        }

    return all_data


if __name__ == "__main__":
    # Get the data
    path_to_data = './data'   # !! change this to your path !!
    path_to_all_data = os.path.join(path_to_data, 'psb', 'all_data.pkl')
    force_recompute = True
    if os.path.exists(path_to_all_data) and not force_recompute:
        with open(path_to_all_data, 'rb') as f:
            all_data = pickle.load(f)
    else:
        database_names = ["GeSiNW_Qubit_VTI01_Jonas_2.db", "GeSiNW_Qubit_VTI01_Jonas_3.db"]
        database_paths = [os.path.join(path_to_data, database_name) for database_name in database_names]

        # Select data from the database
        collected = get_experiment_context(database_paths)

        print('collected is a list with the elements: '
              '(low_magnet_dataset, high_magnet_dataset, '
              'numpy_array_low_magnet, numpy_array_high_magnet, bias_direction)')
        print(f'collected has length {len(collected)}')

        # Get cutouts from the data and save to file
        all_data = get_cutouts_from_collected_data(collected)
        path_to_folder = os.path.join(path_to_data, 'psb')
        if not os.path.exists(path_to_folder):
            os.mkdir(path_to_folder)
        with open(path_to_all_data, "wb") as f:
            pickle.dump(all_data, f)

