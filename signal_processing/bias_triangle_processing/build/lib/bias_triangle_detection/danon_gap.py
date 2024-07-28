import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from typing import Optional
from qcodes.dataset import DataSetProtocol


def extract(data: np.ndarray, sigma: float = 1, field_gap_size: int = 4, relative_depth: float = 1.3) -> Optional[int]:
    """ Extract the gap location from a Danon gap dataset. 
        The gap corresponds to a region along the x axis where the slope is relatively constant.
        The gap is detected by looking at the derivative of the data."""
    smoothed = gaussian(data, sigma)
    sum_abs_derivatives = np.sum(np.abs(np.diff(smoothed, axis=1)), axis=1)
    sum_abs_derivatives = gaussian(sum_abs_derivatives, sigma)
    order_by_derivatives = np.argsort(sum_abs_derivatives)
    y = order_by_derivatives[0]
       # Tolerate only one missing spot in the sequence
    if (np.max(order_by_derivatives[:field_gap_size]) - np.min(order_by_derivatives[:field_gap_size]) > field_gap_size
       # the minima is quite deep wrt to average value
       or np.min(sum_abs_derivatives)*relative_depth > np.median(sum_abs_derivatives)
       # remove extrema
       or y < data.shape[0]//4 or y > data.shape[0]*3//4):
        return None
    return y

def plot_danon(data: np.ndarray, run_id: int, label: bool, pred: bool, y: Optional[int], folder: str):
    plt.clf()
    plt.imshow(data)
    plt.colorbar()
    correct = label == pred
    pred_str = "Positive" if pred else "Negative"
    title = f"{correct}_{pred_str}_{run_id}"
    if pred:
        plt.axhline(y=y, c="red", linestyle = "dashed")
    plt.title(title)
    plt.savefig(folder + title + ".png")
    plt.clf()
    smoothed = gaussian(data)
    sum_derivatives = np.sum(np.abs(np.diff(smoothed, axis=1)), axis=1)
    for i in range(4):
        plt.plot(gaussian(sum_derivatives, sigma=i), label=f"smoothing={i}")
    plt.legend()
    plt.savefig(folder + title + "sum_abs_derivatives.png")

def change_sign(data: np.ndarray, dataset: DataSetProtocol) -> np.ndarray:
    """ According to VSD value change current sign"""
    VSD = dataset.snapshot["station"]["instruments"]["LNHR_dac"]["submodules"]["ch8"]["parameters"]["V_SD"]["value"]
    if VSD is not None and VSD < 0:
        data *= -1
    return data


