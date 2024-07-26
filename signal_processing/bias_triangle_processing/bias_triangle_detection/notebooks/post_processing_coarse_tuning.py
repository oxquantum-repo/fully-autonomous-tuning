from bias_triangle_detection.bayesian_optimization.setup import setup
from functools import reduce
from operator import add
import matplotlib.pyplot as plt
from bias_triangle_detection.qcodes_db_utils import query_datasets
import datetime
import os

def get_tasks_per_measurement(client, measurement_name):
    tasks = []
    for task in client.get_all_tasks():
        if task.user_defined_data is not None and task.user_defined_data["measurement_name"] == measurement_name:
            tasks.append(task)
    return tasks
        
def plot_measurement(dataset, guid_to_result):
    fig, ax = plt.subplots(2, figsize=(7, 9))
    data = dataset.to_xarray_dataset()["I_SD"]
    data.plot(ax = ax[0])
    ax[1].set_axis_off()
    text = str(guid_to_result[dataset.guid])
    ax[1].text(0, 0, text)

def main(local=True):
    if local:
        path = "~/src/nanowire-tuning/data/dummy_db.db"
    else:
        path = "~/Downloads/GeSiNW_Qubit_VTI01_Jonas_2.db"
    # filter only recent runs
    time_ago = {"hours":2, "days":0} 
    time_thresh = (datetime.datetime.now() - datetime.timedelta(**time_ago)).timestamp()
    query = lambda d: d.name.startswith("coarse_tuning") and d.run_timestamp_raw > time_thresh
    datasets = query_datasets(path, query)
    measurement_names = {d.name for d in datasets}
    if local:
        client = setup()
    else:
        client = setup(os.environ["CI_OPTAAS_ADMIN_API_KEY"])
    for measurement_name in measurement_names:
        print("Routine: ", measurement_name)
        tasks = get_tasks_per_measurement(client, measurement_name)
        print("Optimisers: ", len(tasks))
        iterations = [t.number_of_iterations for t in tasks]
        print("Measurements per optimiser: ", iterations)
        results  = reduce(add, [t.get_results() for t in tasks], [])
        assert len(results) == sum(iterations), "Mismatch between number of iterations and number of results"

        guid_to_result = {r.user_defined_data["guid"]: r for r in results}
        measurement_datasets = [d for d in datasets if d.name == measurement_name]
        measurement_datasets
        if local:
            measurement_datasets = [d for d in measurement_datasets if d.guid in guid_to_result]
        else:
            assert len(measurement_datasets) == len(guid_to_result), "Mismatch between measurements and results"
        measurement_datasets.sort(key = lambda d: d.run_timestamp_raw)
        for d in measurement_datasets:
            plot_measurement(d, guid_to_result)
            plt.title(f"\n{d.name}\n in order of acquition")
            plt.show()
        measurement_datasets.sort(key = lambda d: guid_to_result[d.guid].score)
        for d in measurement_datasets:
            plot_measurement(d, guid_to_result)
            plt.title(f"\n{d.name}\n in order of score")
            plt.show()

         
        

if __name__ == "__main__":
    main(local=True)