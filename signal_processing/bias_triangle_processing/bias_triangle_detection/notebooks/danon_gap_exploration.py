import os
from bias_triangle_detection.danon_gap import extract, plot_danon
from bias_triangle_detection.qcodes_db_utils import query_datasets
import numpy as np
import shutil
import os

folder = "danon_gap_exploration/"

shutil.rmtree(folder, ignore_errors=True)
os.makedirs(folder)

path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas.db"

ids = [1025, 1028, 1051, 1054, 1057, 1063, 1065, 1079, 1082, 1085, 1088, 1094, 1101, 1104, 1110, 1115, 1121, 1124,
       1127, 1129, 1132, 1137, 1148, 1151, 1154, 1157, 1160, 1163, 1165, 1168, 1171, 1176, 1180, 1182, 1185, 1191,
       1194, 1202, 1205, 1208, 1211, 1216, 1222]
positive = [1025, 1051, 1065, 1085, 1121, 1124, 1176, 1191, 1202]
is_danon = lambda dataset: dataset.run_id in ids
datasets = query_datasets(path, is_danon)

path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2.db"
ids = [707, 809]
positive +=ids
is_danon = lambda dataset: dataset.run_id in ids
datasets += query_datasets(path, is_danon)

field_gap = 0.02

confusion_matrix = np.zeros((2, 2))
for dataset in datasets:
    data_dict = dataset.get_parameter_data()
    data = dataset.to_xarray_dataset()["I_SD"]
    label = dataset.run_id in positive
    step = np.diff(data[data.dims[0]])[0]
    field_gap_size = int(np.round(field_gap/step)) 
    y = extract(data, field_gap_size = field_gap_size)
    prediction = y is not None
    plot_danon(data, dataset.run_id, label, prediction, y, folder)
    confusion_matrix[int(label), int(prediction)] += 1
print(confusion_matrix)
