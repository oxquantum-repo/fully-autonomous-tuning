import logging
from pathlib import Path

import toml

from helper_functions.data_access_layer import DataAccess
from helper_functions.file_system_tools import create_folder_structure

from base_stages_simple import RootStage
from qubit_characterisation_dummy import RabiOscillations, ResonanceFrequency

from helper_functions.documentation_generator import MarkdownFile

config_path = Path(__file__).parents[1] / "data" / 'config_files' / "example_dummy_config.toml"

# save
configs = toml.load(config_path)
experiment_name = configs["experiment_name"]

create_folder_structure(experiment_name)

md_generator = MarkdownFile(experiment_name)
with open(
        Path(__file__).parents[1] / "data" / "experiments" / experiment_name / "documentation" / "configs.toml", "w"
) as f:
    toml.dump(configs, f)

logging.basicConfig(
    filename=Path(__file__).parents[1] / "data" / "experiments" / experiment_name / "log_files" / "logging.log",
    format="%(asctime)s %(pathname)s line-no:%(lineno)d %(funcName)s %(levelname)s:%(message)s",
    level=logging.INFO,
)

md_generator.create_or_update_file(f"Starting")

data_access_layer = DataAccess(experiment_name, root_folder=Path(__file__).parents[1] / "data" / "experiments")

resonance_frequency = ResonanceFrequency(
    experiment_name,
    configs=configs["ResonanceFrequency"],
    data_access_layer=data_access_layer,
)

rabi_oscillations = RabiOscillations(
    experiment_name,
    configs=configs["RabiOscillations"],
    data_access_layer=data_access_layer,
)

root = RootStage(
    experiment_name,
    configs["Root"],
    data_access_layer,
)

root.child_stages = [resonance_frequency]
resonance_frequency.child_stages = [rabi_oscillations]
rabi_oscillations.child_stages = [resonance_frequency]

print("init done")

root.kick_off()
