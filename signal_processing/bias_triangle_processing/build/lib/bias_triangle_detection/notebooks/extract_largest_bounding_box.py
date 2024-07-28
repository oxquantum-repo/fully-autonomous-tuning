import datetime

import matplotlib.pyplot as plt
import mlflow

from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.bounding_box import scan_to_bounding_boxes
from bias_triangle_detection.switches.fastai_model_wrapper import (
    ArrayGetter,
    FastAISwitchModel,
    LabelGetter,
)

path = "data/GeSiNW_Qubit_VTI01_Jonas_3.db"
path_learner = "models/export.pkl"
path_checkpoint = "best_model_weights"
switch_model = FastAISwitchModel(path_learner, path_checkpoint)
time_ago = {"hours": 2, "days": 50}
time_thresh = (datetime.datetime.now() - datetime.timedelta(**time_ago)).timestamp()
query = (
    lambda d: d.name.startswith("coarse_tuning")
    and not d.metadata["snake_mode"]
    and (d.run_timestamp_raw > time_thresh)
    and (d.to_xarray_dataset()["I_SD"].shape == (100, 100))
)
# query = lambda d: d.guid == "77d4d49d-0000-0000-0000-0189c490c243"

datasets = query_datasets(path, query)
mlflow.set_experiment("bounding_box_extraction")

for qcodes_dataset in datasets:
    dataset = qcodes_dataset.to_xarray_dataset()["I_SD"]
    best_bb,__, fig = scan_to_bounding_boxes(dataset, switch_model)
    file_path = f"data/bounding_boxes_{qcodes_dataset.guid}.png"
    fig.savefig(file_path)
    plt.close(fig)
    mlflow.log_artifact(file_path)
    
