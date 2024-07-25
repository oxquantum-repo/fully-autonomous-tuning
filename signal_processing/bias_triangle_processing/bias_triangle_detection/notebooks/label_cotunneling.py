import matplotlib.pyplot as plt
import numpy as np

from mpl_image_labeller import image_labeller

import matplotlib.pyplot as plt
import numpy as np
from bias_triangle_detection.qcodes_db_utils import query_datasets
import numpy as np
import matplotlib.pyplot as plt
import pickle


path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2 (4).db"
query = lambda dataset: dataset.name.startswith(f"coarse_tuning")
all_datasets = query_datasets(path, query)
images = []
ids = []
print(len(all_datasets))
for dataset in all_datasets:
    image = dataset.to_xarray_dataset()["I_SD"].values
    # skip low res scans
    if image.size < 500:
        continue
    images.append(image)
    ids.append(dataset.run_id)
labeller = image_labeller(
    images,
    classes=["vertical", "horizontal", "none"],
    label_keymap=["a", "s", "z"],
    multiclass=True,
)
plt.show()


ids, onehot = labeller.labels_onehot

assert np.all(np.any(onehot,axis=1))
assert np.all(np.any(onehot[:,:2],axis=1) == ~onehot[:,2])

id_to_label = {id:l[:2] for id, l in zip(ids, onehot)}

description = """ Labelling of the coarse tuning exps with medium to high resolution from the db:
https://drive.google.com/u/0/uc?id=1y8khfFsEqF0nWMczXwXjrEVwhP2rpebS&export=download
dictionary run_id to two boolean labels (vertical_cotunnelling, horizontal_cotunnelling)
"""
with open("labels_dict.pkl", "wb") as f:
    pickle.dump((description, id_to_label), f)