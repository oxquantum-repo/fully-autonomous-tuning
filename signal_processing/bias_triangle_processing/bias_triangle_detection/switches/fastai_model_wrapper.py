import pathlib
import platform
from contextlib import contextmanager

import numpy as np
from fastai.learner import load_learner
from fastai.vision.all import PILImage, SaveModelCallback

from bias_triangle_detection.utils.types import GrayImage, PathLike
from bias_triangle_detection.fastai_training.mlflow_callback import FastAIMLFlowCallback


@contextmanager
def set_posix_windows():
    # see https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
    posix_backup = pathlib.PosixPath
    try:
        if "win" in platform.system().lower():
            pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


# these getters are auxiliary and will be removed. For now are required to load the model
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


class FastAISwitchModel:
    def __init__(self, learner_path: PathLike, model_checkpoint: PathLike, continue_training: bool = False):
        with set_posix_windows():
            self.learner = load_learner(learner_path)
        self.learner.load(model_checkpoint)
        if not continue_training:
            # Remove callbacks even though it is not supposed to use them in prediction
            self.learner.remove_cbs([SaveModelCallback, FastAIMLFlowCallback]),


    def __call__(self, image: GrayImage) -> bool:
        assert image.ndim == 2
        # three identical channels
        image = np.repeat(image[..., None], 3, axis=2)
        image = PILImage.create(image)
        is_good_triangle, _, _ = self.learner.predict(image)
        return is_good_triangle == "no switching"
