import pickle
from typing import Dict, List, Tuple

import numpy as np
import os

# +
import xarray as xr
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    ImageDataLoaders,
    Learner,
    PILImage,
    RandomSplitter,
    Resize,
    aug_transforms,
)


class LabelGetter:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, idx):
        return self.labels[idx]


class ArrayGetter:
    def __init__(self, array):
        self.array = array

    def __call__(self, idx):
        return self.array[idx]


def get_dataloader(
        crops: np.ndarray,
        labels: List[str],
        train_valid_labels: np.ndarray,
        valid_pct: float,
        aug_kwargs: Dict,
) -> ImageDataLoaders:
    splitter = RandomSplitter(valid_pct, seed=None)
    dblock = DataBlock(
        blocks=(ImageBlock(PILImage), CategoryBlock),
        splitter=splitter,
        get_x=ArrayGetter(crops),
        get_y=LabelGetter(labels),
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(**aug_kwargs)],
    )
    return ImageDataLoaders.from_dblock(dblock, train_valid_labels, bs=bs)


def train_model(learn: Learner, n_one_cycle: int, n_unfreeze_one_cycle: int):
    learn.fit_one_cycle(n_one_cycle)

    learn.lr_find()

    learn.unfreeze()
    learn.fit_one_cycle(n_unfreeze_one_cycle, lr_max=slice(1e-6, 1e-4))


def _indices_matching_guid(to_match: str, guid_list: List[str]) -> List[int]:
    return [i for i, guid in enumerate(guid_list) if guid == to_match]


def load_data(path_crops: str, path_labels: str) -> Tuple[np.ndarray, List[str]]:
    crops = xr.open_dataarray(path_crops).to_numpy()
    with open(path_labels, "rb") as f:
        labels = pickle.load(f)
    return crops, labels


def make_crops_and_labels_and_guids_files(file_pdata_directory_pathath: str, labelled_data_dict_filename: str,
                                          crops_filename: str, crops_labels_filename: str, guids_list_filename: str) \
        -> Tuple[str, str, str]:
    with open(os.path.join(data_directory_path, labelled_data_dict_filename), 'rb') as f:
        all_data_dict = pickle.load(f)

    crops_list = []
    labels_list = []
    guids_list = []
    for guid, data_dict in all_data_dict.items():
        crops_list.extend(data_dict["list_of_processed_crops"])
        labels_list.extend(data_dict["list_of_labels"])
        guids_list.extend([guid for i in range(len(data_dict["list_of_processed_crops"]))])

    crops_path, labels_path, guids_path \
        = make_file_paths(data_directory_path, crops_filename, crops_labels_filename, guids_list_filename)
    xr.concat(crops_list, 'crops').to_netcdf(crops_path)
    with open(labels_path, 'wb') as fx:
        pickle.dump(labels_list, fx)
    with open(guids_path, 'wb') as fy:
        pickle.dump(guids_list, fy)

    return crops_path, labels_path, guids_path


def make_file_paths(data_location: str, crops_filename: str, crops_labels_filename: str, guids_list_filename: str) \
        -> Tuple[str, str, str]:
    crops_path = os.path.join(data_location, crops_filename)
    labels_path = os.path.join(data_location, crops_labels_filename)
    guids_path = os.path.join(data_location, guids_list_filename)
    return crops_path, labels_path, guids_path


def check_data_exists_on_file(data_directory_path: str,
                              crops_filename: str, crops_labels_filename: str, guids_list_filename: str) -> bool:
    crops_path, labels_path, guids_path \
        = make_file_paths(data_directory_path, crops_filename, crops_labels_filename, guids_list_filename)
    return os.path.exists(crops_path) and os.path.exists(labels_path) and os.path.exists(guids_path)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np

    # +
    from fastai.vision.all import (
        SaveModelCallback,
        default_device,
        error_rate,
        resnet34,
        set_seed,
        vision_learner,
    )
    from sklearn.model_selection import train_test_split

    from bias_triangle_detection.fastai_training.mlflow_callback import (
        FastAIMLFlowCallback,
    )

    # 1. Set up model and other parameters on experiment tracking system (mlflow)
    experiment_name = "train_fastai_model"
    mlflow.set_experiment(experiment_name)

    default_device(False)
    random_state = 2
    set_seed(random_state)

    mlflow.log_param("random_state", random_state)
    bs = 64
    mlflow.log_param("bs", bs)
    validation_size = 0.15
    mlflow.log_param("validation_size", validation_size)
    test_size = 0.15
    mlflow.log_param("test_size", test_size)

    # 2. Get labeled data for training and testing
    #   - first check if data is on file already, and if not make the required files from the labelled_data_dict
    data_directory_path = "data/switch_detection"
    labelled_data_dict_filename = "data_and_labels_nathan.pkl"
    crops_filename = "crops_train.nc"
    crop_labels_filename = "crop_train_labels.pkl"
    guids_list_filename = "uids_crops.pkl"

    if check_data_exists_on_file(data_directory_path, crops_filename, crop_labels_filename, guids_list_filename):
        print("Data found on file.")
    elif os.path.exists(os.path.join(data_directory_path, labelled_data_dict_filename)):
        make_crops_and_labels_and_guids_files(data_directory_path, labelled_data_dict_filename, crops_filename,
                                              crop_labels_filename, guids_list_filename)
        print("Data not found on file. Files created from labelled data dictionary.")
    else:
        raise FileNotFoundError(f"No files found from which to training data can be loaded.")

    path_crops, path_crop_labels, path_guid_list \
        = make_file_paths(data_directory_path, crops_filename, crop_labels_filename, guids_list_filename)

    #   - load data and check dimensions
    crops, labels = load_data(path_crops, path_crop_labels)
    with open(path_guid_list, 'rb') as fp:
        guid_list = pickle.load(fp)
    assert len(crops) == len(labels), "Number of crops and labels must be equal"
    assert len(crops) == len(guid_list), "Number of guids must equal the number of crops"

    #   - next, create the train/test split minimising guid overlap between train and test sets
    total_n_crops = len(labels)

    indices_train_valid, indices_test = [], []
    all_indices = list(range(len(guid_list)))
    for guid in set(guid_list):
        if len(indices_test) / len(guid_list) < test_size:
            indices_test += _indices_matching_guid(guid, guid_list)
        else:
            indices_train_valid += _indices_matching_guid(guid, guid_list)

    print(
        f"Split test and train grouped by GUID\n N Train/Validation: {len(indices_train_valid)}\n Test: {len(indices_test)}")
    test_indices_path = "data/switch_detection/test_indices.npy"

    with open(test_indices_path, "wb") as f:
        np.save(f, indices_test)

    mlflow.log_artifact(test_indices_path)

    crops = np.repeat(crops[..., None], 3, axis=3)

    aug_transforms_kwargs = {
        "do_flip": True,
        "flip_vert": True,
        "max_rotate": 180,
        "size": 224,
    }
    mlflow.log_params(aug_transforms_kwargs)

    # indices_train_valid size has size (1-test_size) wrt the original size.
    # validation_size is wrt the original size. Below this as pct of indices_train_valid.
    valid_pct = validation_size / (1 - test_size)

    dls = get_dataloader(
        crops, labels, indices_train_valid, valid_pct, aug_transforms_kwargs
    )
    dls.show_batch(max_n=9, figsize=(7, 6))
    mlflow.log_figure(plt.gcf(), "batch_sample.png")

    # 3. Build model
    name_best_model_weights = "best_model_weights"
    learn = vision_learner(
        dls,
        resnet34,
        metrics=error_rate,
        cbs=[
            FastAIMLFlowCallback(),
            SaveModelCallback(fname=name_best_model_weights),
        ],
    )

    learn.export("models/export.pkl")

    train_model(learn, n_one_cycle=20, n_unfreeze_one_cycle=2)