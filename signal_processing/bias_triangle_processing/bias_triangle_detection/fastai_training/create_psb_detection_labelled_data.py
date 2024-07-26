import getpass
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt, patches
from mpl_image_labeller import image_labeller
from tqdm import tqdm

from bias_triangle_detection.fastai_training.utils import make_file_path_from_folder_path_and_guid, get_img_from_fig


def get_images(collected_cutouts: Dict, path: str, guids: str, override: bool = False) \
        -> List[np.ndarray]:
    # If images are not already on file, make them and save them to file
    if not os.path.exists(path):
        os.mkdir(path)
    file_path = make_file_path_from_folder_path_and_guid(path, guids)

    if override is True:
        list_imgs = make_images(collected_cutouts, guids)
        with open(file_path, "wb") as f:
            pickle.dump(list_imgs, f)
    else:
        try:
            with open(file_path, "rb") as f:
                list_imgs = pickle.load(f)
        except:
            list_imgs = make_images(collected_cutouts, guids)
            with open(file_path, "wb") as f:
                pickle.dump(list_imgs, f)

    return list_imgs


def make_images(collected_cutouts: dict, guids: str) -> List[np.ndarray]:
    list_imgs = []

    for cutout, triangle_ctr in zip(collected_cutouts['cutouts'], collected_cutouts['all_triangles_px']):
        if cutout is None:
            continue
        list_imgs.append(get_image(cutout[1], cutout[0],
                                   collected_cutouts['numpy_arrays']['high_magnet'],
                                   collected_cutouts['numpy_arrays']['low_magnet'],
                                   triangle_ctr, guids))
    return list_imgs


def get_image(cutout_high: np.ndarray, cutout_low: np.ndarray, original_high: np.ndarray, original_low: np.ndarray,
              triangle_ctr: Tuple, guid: str) -> np.ndarray:
    # Set up the basic plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0][0].imshow(cutout_high, origin='lower')
    axs[0][0].set_title('high magnet')
    axs[0][1].imshow(cutout_low, origin='lower')
    axs[0][1].set_title('low magnet')
    axs[1][0].imshow(original_high)
    axs[1][0].set_title('high magnet wide shot')
    axs[1][1].imshow(original_low)
    axs[1][1].set_title('low magnet wide shot')
    for ax in axs.flatten():
        ax.axis('off')
    # Create a Rectangle patch to show how the cutouts relate to the originals
    side_length = cutout_high.shape[0]
    middle = triangle_ctr
    start_hor = middle[1] - np.floor(side_length / 2)
    start_vert = middle[0] - np.floor(side_length / 2)
    rect1 = patches.Rectangle((start_hor, start_vert), side_length, side_length,
                              linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((start_hor, start_vert), side_length, side_length,
                              linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the originals' axes
    axs[1][0].add_patch(rect1)
    axs[1][1].add_patch(rect2)
    plt.suptitle(guid)
    plt.tight_layout()

    # Get image instead of the figure, and close the figure
    img = get_img_from_fig(fig)
    plt.close(fig)

    return img


if __name__ == "__main__":
    # Make psb labels for cropped triangle data created by create_psb_detection_training_data.py
    #   1. Get the data
    path_to_data = './data/psb'   # !! change this to your path !!
    path_to_all_data = os.path.join(path_to_data, 'all_data.pkl')
    force_recompute = False
    if os.path.exists(path_to_all_data):
        with open(path_to_all_data, 'rb') as f:
            all_data = pickle.load(f)
    else:
        FileNotFoundError(f'No file at {path_to_all_data}')

    #   2. Create or get crops for labelling from the measurement scans and make figures showing cropped triangle and
    #   context
    override = False  # Set to False if you want to pick up the data directly from file. Set to True if you want to
                      # recreate these figures from scratch.
    print("Creating images showing image to label and context for that image:")
    images_for_labelling = []
    path_images = "data/psb/imgs_to_label"
    for guid, collected_cutouts in tqdm(all_data.items()):
        images_for_labelling += get_images(collected_cutouts, path_images, guid,
                                           override=override)

    #   3. Get the user to label images.
    labeller = image_labeller(
        images_for_labelling,
        classes=["psb", "no psb", "junk"],
        label_keymap=["j", "k", "l"],
        title='Image Index: {image_index}/' + f'{len(images_for_labelling) - 1}'
    )
    plt.show()

    labels = labeller.labels
    for guid, collected_cutouts in all_data.items():
        n_crops = len(collected_cutouts["cutouts"])
        all_data[guid]["list_of_labels"] = labels[:n_crops]
        labels = labels[n_crops:]

    #   4. Write labelled data to file as a pickle
    user = getpass.getuser()
    user = user.lower().replace(" ", "_")
    labels_path = f"data/psb/data_and_labels_{user}.pkl"

    #   5. Save labels to file
    with open(labels_path, "wb") as f:
        pickle.dump(all_data, f)