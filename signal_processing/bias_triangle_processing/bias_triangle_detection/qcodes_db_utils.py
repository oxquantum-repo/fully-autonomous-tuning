import os
from typing import Callable, List

from qcodes.dataset import (
    DataSetProtocol,
    experiments,
    initialise_or_create_database_at,
    load_by_guid,
    plot_dataset,
)


def check_condition(
    datasets: List[DataSetProtocol], condition: Callable[[DataSetProtocol], bool]
) -> List[str]:
    guids = []
    for dataset in datasets:
        try:
            if condition(dataset):
                guids.append(dataset.guid)
        except Exception as e:
            continue
    return guids


def get_datasets_from_guids(guids: List[str]) -> List[DataSetProtocol]:
    datasets = []
    for guid in guids:
        try:
            dataset = load_by_guid(guid)
            datasets.append(dataset)
        except Exception as e:
            continue
    return datasets


def query_datasets(path: str, query: Callable) -> List[DataSetProtocol]:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Database file does not exist {path}")
    initialise_or_create_database_at(path)
    list_of_experiments = experiments()
    
    datasets = sum((exp.data_sets() for exp in list_of_experiments), [])

    guids = check_condition(datasets, query)

    return get_datasets_from_guids(guids)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas.db"

    is_1025 = lambda dataset: dataset.run_id == 1025
    datasets = query_datasets(path, is_1025)

    for d in datasets:
        axs, cbs = plot_dataset(d)
        plt.show()

