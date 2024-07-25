import pickle
import os
import logging


def create_folder_structure(experiment_name, root_folder="../data/experiments/"):
    logger_folder = os.path.join(root_folder, experiment_name, "log_files")
    figures_folder = os.path.join(root_folder, experiment_name, "figures")
    extracted_data_folder = os.path.join(root_folder, experiment_name, "extracted_data")
    documentation_folder = os.path.join(root_folder, experiment_name, "documentation")
    node_states_folder = os.path.join(root_folder, experiment_name, "node_states")
    print(f"Creating folder structure for experiment '{experiment_name}'")
    print(f"Root folder: {root_folder}")
    print(f"Logger folder: {logger_folder}")

    os.makedirs(logger_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)
    os.makedirs(extracted_data_folder, exist_ok=True)
    os.makedirs(documentation_folder, exist_ok=True)
    os.makedirs(node_states_folder, exist_ok=True)

