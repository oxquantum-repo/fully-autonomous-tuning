import imgaug.augmenters as iaa
import numpy as np
import math

from multiprocessing import Pool

from torchvision.transforms import ColorJitter
from PIL import Image

from skimage.util import random_noise

import matplotlib.pyplot as plt

from itertools import repeat


def wrapper_augment_single(args):
    """Needed for multiprocessing as it can't handle starred expressions"""
    return augment_single(*args)


def augment_single(
    img: np.array, shear_and_stretch: bool, sampling_func: callable
) -> np.array:
    """Apply the augmentations to a single element.

    Args:
        img: Image to be augmented. Expects a (height x width) array.
        shear_and_stretch: Whether to perform shearing and stretching or not
        sampling_func: Function that returns the augmentation factors
    Returns:
        Augmented and normalised image. Same size as input image but may be cropped.
    """
    resizer = iaa.Resize(img.shape[-2:])
    augmentation_params = sampling_func()

    img_processed = normalise(img)
    img_processed = process(
        img_processed, *augmentation_params, shear_and_stretch=shear_and_stretch
    )
    img_processed = np.array(img_processed)
    img_processed = resizer.augment_image(image=img_processed)
    img_processed = normalise(img_processed)
    return img_processed


def augment_batch(
    array_of_imgs: np.array,
    sampling_func: callable,
    shear_and_stretch: bool = True,
    multiprocessing_on: bool = False,
    n_workers: int = 1,
) -> np.array:
    """Apply an augmentation to each image given.

    Args:
        array_of_imgs: Images to be augmented. Expects a (n_images x height x width) array.
        sampling_func: Function that returns the augmentation factors
        shear_and_stretch: Whether to perform shearing and stretching or not
        multiprocessing_on: Whether to use multiprocessing or not
        n_workers: Number of processes if multiprocessing is turned on.
    Returns:
        Augmented and normalised images. Same size as input array.
    """
    if multiprocessing_on:
        p = Pool(processes=n_workers)
        new_array_of_imgs = p.map(
            wrapper_augment_single,
            [(el, shear_and_stretch, sampling_func) for el in array_of_imgs],
        )
        p.close()
        p.join()
        return np.array(new_array_of_imgs)

    new_array_of_imgs = []
    for img in array_of_imgs:
        img_processed = augment_single(img, shear_and_stretch, sampling_func)
        new_array_of_imgs.append(img_processed)
    return np.array(new_array_of_imgs)


def adjust_brightness(img: np.array, brightness: float) -> np.array:
    """Change brightness of a given image.

    Args
        img: Should be a single image, normalised between 0 and 1
        brightness: Float how much to change the image. 0 changes nothing.
    """
    return np.clip(img + brightness, 0, 1)


def adjust_contrast(img: np.array, contrast: float) -> np.array:
    """Change contrast of a given image.

    Args
        img: Should be a single image, normalised between 0 and 1
        contrast: Float how much to change the image. 0 changes nothing.
    """
    mean = np.mean(img)
    img = (img - mean) * contrast + mean
    return np.clip(img, 0, 1)


def random_crop(image: np.array, output_size: float = None):
    """Randomly crop a given image.

    Will crop the image at a random point to 3/4 of the original size
    or the size given by "output_size"

    Args
        img: Should be a single image, normalised between 0 and 1
        output_size: If you want a certain image size, you can set this
                     optional parameter
    """
    h, w = image.shape[-2:]
    if output_size == None:
        new_h, new_w = (
            int(3 * h / 4),
            int(3 * w / 4),
        )
    else:
        new_h, new_w = int(output_size * h), int(output_size * w)

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top : top + new_h, left : left + new_w]
    return image


def normalise(img: np.array, maximum: float = None, minimum: float = None) -> np.array:
    """Perform normalisation of a single given image.

    If given max and min values, use those for normalisation.

    Args
        img: Image to be normalised
        maximum: Optional, maximal value to used in normalisation
        minimum: Optional, minimal value to used in normalisation
    """
    img = np.asarray(img)
    if maximum == None:
        maximum = np.max(img)
    if minimum == None:
        minimum = np.min(img)
    return (img - minimum) / (maximum - minimum)


def sample_simple_augmentation_factors(
    correct_bias=False,
    flip_chance=0.5,
    max_shearing_angle_horizontal=20,
    max_shearing_angle_vertical=20,
    max_stretching_ratio_x=0.2,
    max_stretching_ratio_y=0.2,
    brightness_range=[-0.2, 0.2],
    contrast_range=[0.8, 1.2],
    max_noise_level=0.003,
    cropping_range=[0.7, 1.0],
    visualise=False,
):
    """Sample augmentation factors.

    Args:
        correct_bias: If you want, e.g., bias triangles to all appear as if they have
                      the same bias direction, use this keyword.
        flip_chance: With this probability, flip along the diagonal. If you want to
                     test this, a flip will turn ```img``` into  ```img[::-1, ::-1].T```
        max_shearing_angle_horizontal: Maximum shearing angle in degrees for a horizontal shear
        max_shearing_angle_vertical: Maximum shearing angle in degrees for a vertical shear
        max_stretching_ratio_x: Maximum stretch factor for a horizontal stretch
        max_stretching_ratio_y: Maximum stretch factor for a vertical stretch
        brightness_range: Sampling range for brightness adjustment
        contrast_range: Sampling range for contrast adjustment
        max_noise_level: Gaussian noise of each pixel
        cropping_range: Window size of random crop as fraction of original size
        visualise: Visualises the augmentation.
    """
    np.random.seed()

    shearing_angle_h = (np.random.uniform() * 2 - 1) * max_shearing_angle_horizontal
    shearing_angle_v = (np.random.uniform() * 2 - 1) * max_shearing_angle_vertical
    stretching_ratio_x = (np.random.uniform() * 2 - 1) * max_stretching_ratio_x
    stretching_ratio_y = (np.random.uniform() * 2 - 1) * max_stretching_ratio_y
    cropping_size = np.random.uniform() * (
        np.max(cropping_range) - np.min(cropping_range)
    ) + np.min(cropping_range)
    if visualise:
        print(
            "shear h, shear v, stretch x, stretch y",
            shearing_angle_h,
            shearing_angle_v,
            stretching_ratio_x,
            stretching_ratio_y,
        )

    flipped = np.random.uniform() < flip_chance
    brightness = np.random.uniform() * (
        np.max(brightness_range) - np.min(brightness_range)
    ) + np.min(brightness_range)
    contrast = np.random.uniform() * (
        np.max(contrast_range) - np.min(contrast_range)
    ) + np.min(contrast_range)
    noise = np.random.uniform() * np.max(max_noise_level)
    return (
        shearing_angle_h,
        shearing_angle_v,
        stretching_ratio_x,
        stretching_ratio_y,
        flipped,
        brightness,
        contrast,
        noise,
        cropping_size,
    )


def process(
    img,
    shearing_angle_h,
    shearing_angle_v,
    stretching_ratio_x,
    stretching_ratio_y,
    flipped,
    brightness,
    contrast,
    noise,
    cropping_size,
    correct_bias=False,
    shear_and_stretch=True,
    visualise=False,
):
    """Perform augmentation on an image."""
    if correct_bias:
        img = img.T * -1
        img = img[::-1, ::-1].T

    if visualise:
        plt.imshow(img)
        plt.show()

    if shear_and_stretch:
        img = iaa.geometric.Affine(
            scale={"x": 1 + stretching_ratio_x, "y": 1 + stretching_ratio_y}
        ).augment_image(img)
        if visualise:
            plt.imshow(img)
            plt.show()
        # shear horizontally
        img = iaa.geometric.Affine(shear=shearing_angle_h, order=5).augment_image(img)
        if visualise:
            plt.imshow(img)
            plt.show()
        # shear vertically
        img = np.rot90(img)
        img = iaa.geometric.Affine(shear=shearing_angle_v, order=5).augment_image(img)
        img = np.rot90(img, k=3)
        if visualise:
            plt.imshow(img)
            plt.show()
    if flipped:
        img = img[::-1, ::-1].T
    if visualise:
        plt.imshow(img)
        plt.show()
    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = random_noise(img, var=noise)
    img = random_crop(img, cropping_size)
    return img
