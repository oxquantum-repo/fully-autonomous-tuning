import io
import os

import cv2
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt


def get_img_from_fig(fig: matplotlib.figure.Figure, dpi: int = 180) \
        -> np.ndarray:
    """
    Convert a matplotlib figure into an image in the form of an np.ndarray.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.close(fig)
    return img


def make_file_path_from_folder_path_and_guid(path: str, guid: str) \
        -> str:
    path = os.path.join(path, f'imgs_for_{guid}.pkl')
    return path