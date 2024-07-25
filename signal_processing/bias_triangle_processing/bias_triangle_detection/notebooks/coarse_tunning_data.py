"""Basic script to save the data in a compatible format"""
import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from bias_triangle_detection.im_utils import img_resize_gray
from bias_triangle_detection.utils.xarray_and_qcodes import vector_func


def sanitise(path) -> str:
    return str(path).replace("/", "-").replace(".", "__")


def main() -> xr.DataArray:
    data_path = "data/stab_diagrams"
    rank = 0
    label_attribute = "is_triangle"
    shape = (150, 150)
    dataset = {}
    _func = partial(img_resize_gray, shape=shape)
    func = lambda arr: _func(-1 * arr)
    data_var = "I_SD"
    for filename in tqdm(Path(data_path).rglob("*.pkl")):
        dic = pickle.load(open(filename, "rb"))
        data = dic["xarray"][data_var]
        assert data.shape == (150, 150)
        data = vector_func(data, func)
        data.attrs[label_attribute] = 1
        rank += 1
        data.assign_coords({"rank": rank})
        dataset[sanitise(filename)] = data

    reference_xarray = data.copy()

    root_dir = "data/ist3_data"
    for folder in os.scandir(root_dir):
        print(folder)
        for run_folder in os.scandir(os.path.join(folder, "data")):
            print(run_folder)
            try:
                filename = os.path.join(run_folder, "current_maps.npy")
                d = np.load(filename, allow_pickle=True)
                # assign coords from reference_xarray and datavars
                data = xr.DataArray(
                    d[-1]["ultra_high_res_scan"],
                    dims=reference_xarray.dims,
                    name=data_var,
                    attrs=reference_xarray.attrs,
                )
                data = vector_func(data, func)
                data.attrs[label_attribute] = 0
                rank += 1
                data.assign_coords({"rank": rank})
                dataset[sanitise(filename)] = data

            except:
                print("couldn't load")

    # full_data = xr.concat(dataset.values(), dim="rank")
    dataset = xr.Dataset(dataset)

    return dataset


if __name__ == "__main__":
    full_data = main()
    full_data.to_netcdf("data/coarse_tunning_data.nc")
    # full_data[-9:].plot(
    #     x=full_data.dims[1], y=full_data.dims[2], col=full_data.dims[0], col_wrap=3
    # )
