import matplotlib.pyplot as plt
from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.cotunnelling_detection import adaptive_threshold, is_1d_cotunneling, get_parallel_axis_condition, is_2d_cotunneling, get_cotunneling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

from bias_triangle_detection.qcodes_db_utils import query_datasets

import pickle
import mlflow

def main(plotting=False):
    experiment_name = "cotunneling"
    mlflow.set_experiment(experiment_name)
    with open("labels_log_dict.pkl", "rb") as f:
        _, id_to_labels = pickle.load(f)
    path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2.db"
    query = lambda dataset: dataset.run_id in id_to_labels
    datasets = query_datasets(path, query)
    labels = []
    predictions = []
    for dataset in datasets:
        dataset = dataset.to_xarray_dataset()
        data = dataset["I_SD"]
        thresh = adaptive_threshold(data)
        masks, cotunneling_x_and_y = is_1d_cotunneling(thresh, return_mask=True)
        mask_x, mask_y = masks
        cotunneling_x, cotunneling_y = cotunneling_x_and_y
        mask_x_y, cotunneling_x_y = is_2d_cotunneling(thresh, return_mask=True)
        predictions.append([cotunneling_x or cotunneling_x_y, cotunneling_y or cotunneling_x_y, cotunneling_x_y])
        label_y, label_x = id_to_labels[dataset.run_id]
        label_x_y = label_x and label_y
        labels.append([label_x, label_y, label_x_y])
        false_positive = (not label_x and cotunneling_x or
                          not label_y and cotunneling_y or
                          not label_x_y and cotunneling_x_y)        
        if false_positive:
            data_inv = -data
            log = np.log(data_inv - np.min(data_inv))
            log = np.clip(log, np.percentile(log, 0.1), np.percentile(log, 99.9))
            fig, ax = plt.subplots(3, 2, figsize=(5, 10))
            data.plot(ax=ax[0,0], yincrease=False)
            ax[0,0].set_title('Original')
            # flipping sign as `inv` is set to True
            ax[2,0].set_title("Log")
            ax[1,0].imshow(log)
            ax[2,0].set_title("Mask")
            ax[2,0].imshow(thresh)
            ax[0,1].imshow(mask_x)
            ax[0,1].set_title(f'{cotunneling_x=}')
            ax[1,1].imshow(mask_y)
            ax[1,1].set_title(f'{cotunneling_y=}')
            ax[2,1].imshow(mask_x_y)
            ax[2,1].set_title(f'{cotunneling_x_y=}')
        
            prediction_string = f"{int(cotunneling_x==label_x)}{int(cotunneling_y==label_y)}{int(cotunneling_x_y==label_x_y)}"

            filepath = f"data/{prediction_string}_{dataset.run_id}.png"
            plt.savefig(filepath)
            mlflow.log_artifact(filepath)

    predictions = np.array(predictions)
    labels = np.array(labels)

    f1 = f1_score(labels, predictions, average=None)
    print(f1)
    types = ["x", "y", "x_y"]
    mlflow.log_metrics({f"f1_{t}": v for v, t in zip(f1, types)})
    for i, type in enumerate(types):
        cf = confusion_matrix(labels[:,i], predictions[:,i])
        print(type)
        print(cf)
        mlflow.log_metrics({f"tn_{type}": cf[0,0], f"tp_{type}": cf[1,1], f"fp_{type}": cf[0,1], f"fn_{type}": cf[1,0]})

if __name__ == "__main__":
    main(False)