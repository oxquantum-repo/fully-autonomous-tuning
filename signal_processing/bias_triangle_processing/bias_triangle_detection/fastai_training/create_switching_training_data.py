import os.path
import pickle

from qcodes.dataset import DataSetProtocol
from tqdm import tqdm
from typing import List, Dict

from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.switches.characterisation import get_masks_as_xr


def get_data_from_file(path: str) \
        -> List[DataSetProtocol]:
    query = lambda d: d.name.startswith("coarse_tuning") and (
            d.to_xarray_dataset()["I_SD"].shape == (100, 100)
    )

    datasets = query_datasets(path, query)
    return datasets


def make_training_data_from_file(path: str) \
        -> Dict:
    datasets = get_data_from_file(path)

    all_data = {}
    print("Creating crops from masks of measurement scans:")
    for qcodes_dataset in tqdm(datasets):
        # Get measurement data; Create masks, resized crops, and indices of original crops
        dataset = qcodes_dataset.to_xarray_dataset()["I_SD"]
        try:
            masks, axes_dims = get_masks_as_xr(dataset)
            (
                filtered_candidates_mask,
                candidate_mask,
                corner_pts_mask,
                switch_mask,
                outlier_mask,
                gray_diff,
                crops,
                crops_idcs
            ) = masks
        except:
            print(f"Issues in extracting masks from dataset {qcodes_dataset.guid}")
            continue

        # Record "guid" for future plotting and add to containers

        guid = qcodes_dataset.guid

        crops.attrs["guid"] = guid
        dataset.attrs["guid"] = guid
        crops_idcs.attrs["guid"] = guid

        all_data[guid] = {"measurement_scan_data": dataset,
                          "list_of_processed_crops": crops,
                          "list_of_original_crops": crops_idcs}

    return all_data


if __name__ == "__main__":
    # Before running this file, you need to put the data into the location path and write its name
    data_directory_path = "./data"
    data_filename = "GeSiNW_Qubit_VTI01_Jonas_3.db"

    data_location = os.path.join(data_directory_path, data_filename)
    if os.path.exists(data_location):
        all_data = make_training_data_from_file(data_location)
    else:
        if not os.path.exists(data_directory_path):
            os.mkdir(data_directory_path)
            raise NotADirectoryError(f"No data directory {data_directory_path}. "
                                     f"Please put the data in this location.")
        else:
            raise FileNotFoundError(f"No file found at {data_location}. "
                                    "Please check data has been put in this location.")

    # Write data to file as pickles
    file_name = "crops_and_measurements_dict.pkl"
    path_to_folder = os.path.join(data_directory_path, 'switch_detection')
    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)
    path_crop_and_measurements_dict = os.path.join(path_to_folder, file_name)

    with open(path_crop_and_measurements_dict, "wb") as f:
        pickle.dump(all_data, f)
