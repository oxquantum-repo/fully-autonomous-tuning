from bias_triangle_detection.switches.characterisation import get_masks_as_xr
from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.switches.fastai_model_wrapper import (
    ArrayGetter,
    FastAISwitchModel,
    LabelGetter,
)
from time import time

path = "data/GeSiNW_Qubit_VTI01_Jonas_3.db"
path_learner = "models/export.pkl"
path_checkpoint = "best_model_weights"

now = time()
print("Load model")
switch_model = FastAISwitchModel(path_learner, path_checkpoint)
print("Model loaded in", time() - now, "s")
query = lambda d: d.guid == "77d4d49d-0000-0000-0000-0189c490c243"

dataset = query_datasets(path, query)[0]

full_scan = dataset.to_xarray_dataset()["I_SD"]

now = time()
print("Predict")
get_masks_as_xr(full_scan, switch_model=switch_model)
print("Prediction made in", time() - now, "s")