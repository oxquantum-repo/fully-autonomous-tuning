import matplotlib.pyplot as plt
from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.scores.base import extract_triangles
from bias_triangle_detection.scores.separation import _score_separation

from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.cotunnelling_detection import get_cotunneling
from bias_triangle_detection.switches.characterisation import get_masks_as_xr
from bias_triangle_detection.utils.contours_and_masks import get_individual_components
from bias_triangle_detection.utils.numpy_to_image import to_gray
from bias_triangle_detection.utils.xarray_and_qcodes import xr_to_mask
import numpy as np

import os
import shutil

folder = "data_score_separation/"

def main():
    path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2_7.db"
    #query = lambda dataset: dataset.name.startswith("psb_opti")
    query = lambda dataset: dataset.run_id >= 1766
    datasets = query_datasets(path, query)
    for dataset in datasets:
        dataset = dataset.to_xarray_dataset()
        data = dataset["I_SD"]
        # Fix double inversion due to bug in efficient sampler
        data *= -1
        if data.size > 1000:
            if not any(get_cotunneling(data)):
                score(data, dataset.run_id)
            else:
                print("cotunelling")


def score(im1, run_id, plot=True):
    triangles_mask = get_masks_as_xr(im1)[0][0]
    masks = get_individual_components(xr_to_mask(triangles_mask))[0]
    gray_orig = to_gray(im1)
    print(len(masks))
    for i, mask in enumerate(masks):
        score, z, peaks, min_loc = _score_separation(-gray_orig, mask)
        if plot:
            # x0_m = np.mean([short_side[0][0], short_side[1][0]])
            # y0_m = np.mean([short_side[0][1], short_side[1][1]])
            # x1_m, y1_m = np.mean(base, axis=0)
            fig, axes = plt.subplots(nrows=3)
            fig, axes = plt.subplot_mosaic("ABCD;EEEE")
            axes["A"].imshow(im1)
            axes["B"].imshow(triangles_mask)
            axes["C"].imshow(mask)
            triangle = gray_orig*mask
            x, y = np.nonzero(triangle)
            xl,xr = x.min(),x.max()
            yl,yr = y.min(),y.max()
            axes["D"].imshow(triangle[xl:xr+1, yl:yr+1])
            # axes[0].plot([x0_m, x1_m], [y0_m, y1_m], 'ro-')

            if z is not None:
                axes["E"].plot(z)
                for peak in peaks:
                    axes["E"].plot(peak, z[peak], "x", c="r")
            if min_loc:
                axes["E"].plot(min_loc, z[min_loc], "o", c="r")
            fig.suptitle(f"score_{score:06.2f}")
            plt.savefig(f"{folder}/score_{score:06.2f}_{run_id}_{i}.png")
            plt.clf()

if __name__ == "__main__":

    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    main()