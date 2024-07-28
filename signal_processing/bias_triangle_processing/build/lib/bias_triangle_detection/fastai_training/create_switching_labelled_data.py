from typing import List
import matplotlib.pyplot as plt
import getpass
import pickle
import os
from tqdm import tqdm

from mpl_image_labeller import image_labeller

import numpy as np
import matplotlib.patches as patches
import xarray as xr

from bias_triangle_detection.fastai_training.utils import get_img_from_fig, make_file_path_from_folder_path_and_guid


def get_images(crops_resized: List[xr.DataArray], crops_original_idcs: List[xr.DataArray],
               measurement_scan: xr.DataArray, path: str, guid: str, override: bool = False) \
        -> List[np.ndarray]:
    # If images are not already on file, make them and save them to file
    if not os.path.exists(path):
        os.mkdir(path)
    file_path = make_file_path_from_folder_path_and_guid(path, guid)

    if override is True:
        list_imgs = make_images(crops_resized, crops_original_idcs, measurement_scan)
        with open(file_path, "wb") as f:
            pickle.dump(list_imgs, f)
    else:
        try:
            with open(file_path, "rb") as f:
                list_imgs = pickle.load(f)
        except:
            list_imgs = make_images(crops_resized, crops_original_idcs, measurement_scan)
            with open(file_path, "wb") as f:
                pickle.dump(list_imgs, f)

    return list_imgs


def make_images(crops_resized: List[xr.DataArray], crops_original_idcs: List[xr.DataArray],
                measurement_scan: xr.DataArray) \
        -> List[np.ndarray]:
    list_imgs = []
    for i in range(len(crops_resized)):
        list_imgs.append(get_image(crops_resized[i], crops_original_idcs[i], measurement_scan))

    return list_imgs


def get_image(crop_resized: xr.DataArray, crop_original_idcs: xr.DataArray, measurement_scan: xr.DataArray) \
        -> np.ndarray:
    """
    Create the figure to show to the user when labelling, including the context.
    """
    fig = plt.figure(figsize=(6, 9))
    grid = plt.GridSpec(6, 4)
    ax_resized = plt.subplot(grid[0:2, 0:2])
    ax_original = plt.subplot(grid[0:2, 2:4])
    ax_measurement = plt.subplot(grid[2:6, 0:4])

    ax_resized.matshow(crop_resized)
    ax_resized.set_title('Resized copped image', fontweight='bold')

    # NOTE: The indicies stored in crop_original_idcs are for a mask whose resolution has been increased by a factor of
    # 4. This is done in the function btriangle_detection.py > triangle_segmentation_alg(). Specifically, to create the
    # mask the following function from characterisation.py is called:
    #   extract_candidate_triangle_masks(..., res_h=4, ...)
    # where res_h is set to 4. To get indices for the original image we need to devide crop_original_idcs by res_h.
    start_vert, start_hor, end_vert, end_hor = np.floor(crop_original_idcs.to_numpy() / 4).astype(int)
    ax_original.matshow(measurement_scan[start_vert: end_vert + 1, start_hor: end_hor + 1])
    ax_original.set_title('Original cropped image', fontweight='bold')

    ax_measurement.matshow(measurement_scan)
    ax_measurement.set_title('Crop shown within the original measurement taken', fontweight='bold')

    # Create a Rectangle patch
    rect = patches.Rectangle((start_hor, start_vert), end_hor - start_hor, end_vert - start_vert,
                             linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax_measurement.add_patch(rect)

    # Make axes invisible on all plots
    for ax in [ax_measurement, ax_resized, ax_original]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()

    # Convert figure to an image file
    img = get_img_from_fig(fig)
    plt.close(fig)

    return img


if __name__ == "__main__":
    # Make switching labels for cropped triangle data created by create_switching_training_data.py
    #   1. Get the data
    path_crop_and_measurements = "data/switch_detection/crops_and_measurements_dict.pkl"

    if os.path.exists(path_crop_and_measurements):
        with open(path_crop_and_measurements, "rb") as f:
            all_data = pickle.load(f)
    else:
        FileNotFoundError(f'No file at {path_crop_and_measurements}')

    #   2. Create or get crops for labelling from the measurement scans and make figures showing cropped triangle and
    #   context
    override = False  # Set to False if you want to pick up the data directly from file. Set to True if you want to
                      # recreate these figures from scratch.
    print("Creating images showing image to label and context for that image:")
    images_for_labelling = []
    path_images = "data/switch_detection/imgs_to_label"
    for guid, dataset_data in tqdm(all_data.items()):
        # Get data for this measurement scan and make figures showing cropped triangle and context
        measurement_scan = dataset_data["measurement_scan_data"]
        crops_resized = dataset_data["list_of_processed_crops"]
        crops_original_idcs = dataset_data["list_of_original_crops"]
        images_for_labelling += get_images(crops_resized, crops_original_idcs, measurement_scan, path_images, guid,
                                           override=override)

    #   3. Get the user to label images.
    labeller = image_labeller(
        images_for_labelling,
        classes=["clear switching", "some switching", "no switching"],
        label_keymap=["j", "k", "l"],
        title='Image Index: {image_index}/' + f'{len(images_for_labelling) - 1}'
    )
    plt.show()

    labels = labeller.labels
    for guid, dataset_data in all_data.items():
        n_crops = len(dataset_data["list_of_original_crops"])
        all_data[guid]["list_of_labels"] = labels[:n_crops]
        labels = labels[n_crops:]

    #   4. Write labelled data to file as a pickle
    user = getpass.getuser()
    user = user.lower().replace(" ", "_")
    labels_path = f"data/switch_detection/data_and_labels_{user}.pkl"

    #   5. Save labels to file
    with open(labels_path, "wb") as f:
        pickle.dump(all_data, f)
