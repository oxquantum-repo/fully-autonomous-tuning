from functools import partial
from typing import Callable, List

from qcodes.dataset import (
    DataSetProtocol,
    experiments,
    initialise_or_create_database_at,
    load_by_guid,
)

from bias_triangle_detection.scores.avg_number_of_edges import score_number_of_edges


def check_condition(
    datasets: List[DataSetProtocol], condition: Callable[[DataSetProtocol], bool]
) -> List[str]:
    guids = []
    for dataset in datasets:
        try:
            if condition(dataset):
                print(dataset)
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


def has_name(dataset: DataSetProtocol, *, name: str) -> bool:
    return name in dataset.name


if __name__ == "__main__":
    path = "data/GeSiNW_Qubit_VTI01_Jonas_2.db"

    initialise_or_create_database_at(path)

    list_of_experiments = experiments()

    # there is only one exp
    experiment = list_of_experiments[0]

    # about 1200 datasets. Tipically one dataset is one scan
    datasets = experiment.data_sets()

    # get guids of datasets in coarse tuning runs
    guids_coarse_tuning = check_condition(
        datasets, partial(has_name, name="coarse_tuning")
    )

    datasets_coarse_tuning = get_datasets_from_guids(guids_coarse_tuning)
    for dataset in datasets_coarse_tuning:
        readout_name = dataset.dependent_parameters[0].name
        numpy_array = dataset.get_parameter_data(readout_name)[readout_name][
            readout_name
        ]
        if len(numpy_array) == 0:
            continue
        score = score_number_of_edges(numpy_array)
        print(score)
