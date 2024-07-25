import pickle
from functools import cmp_to_key

import matplotlib.pyplot as plt
import numpy as np
from mpl_image_labeller import image_labeller
from qcodes.dataset import DataSetProtocol, initialise_or_create_database_at

from bias_triangle_detection.qcodes_db_utils import get_datasets_from_guids


def comparison(ds1: DataSetProtocol, ds2: DataSetProtocol) -> int:
    key = "I_SD"
    arr1 = ds1.get_parameter_data(key)[key][key]
    arr2 = ds2.get_parameter_data(key)[key][key]
    arr = np.concatenate([arr1, arr2], axis=1)[None, :, :]
    labeller = image_labeller(
        arr,
        classes=["left", "right", "same"],
        label_keymap=["j", "k", "l"],
        multiclass=False,
    )
    labeller.on_label_assigned(lambda *args, **kw: plt.close())
    plt.show()
    if labeller.labels[0] == "left":
        return 1
    if labeller.labels[0] == "right":
        return -1
    return 0


if __name__ == "__main__":
    path = "data/GeSiNW_Qubit_VTI01_Jonas_2.db"
    initialise_or_create_database_at(path)
    guids_random_subsample = [
        "a028bbc3-0000-0000-0000-01893c6e5860",
        "9a141f4a-0000-0000-0000-01893df02225",
        "ffdd808f-0000-0000-0000-01893cd39432",
        "44372194-0000-0000-0000-01893a86d188",
        "2e1610bd-0000-0000-0000-01893bd7e6ee",
        "4d35b21f-0000-0000-0000-01893d02f4c4",
        "7665e21a-0000-0000-0000-01893d61d8ee",
        "0f872029-0000-0000-0000-01893c0745fc",
        "0eb36b7a-0000-0000-0000-01893ae5828d",
        "ab7d076e-0000-0000-0000-01893ca2b22f",
        "e72ee131-0000-0000-0000-01893f14b513",
        "08891031-0000-0000-0000-01893ab62313",
    ]
    subsample = get_datasets_from_guids(guids_random_subsample)
    sorted_subsample = sorted(subsample, key=cmp_to_key(comparison))
    sorted_guids = [ds.guid for ds in sorted_subsample]
    path_sorted_guids = "data/sorted_guids.pkl"
    with open(path_sorted_guids, "wb") as f:
        pickle.dump(sorted_guids, f)
