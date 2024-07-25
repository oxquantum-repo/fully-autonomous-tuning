import logging

import matplotlib.pyplot as plt
import numpy as np
import toml
import torchvision
from bias_triangle_detection.switches.fastai_model_wrapper import ArrayGetter, LabelGetter


from pipelines.base_stages import RootStage, TerminalStage, RootStageFromScratch
from pipelines.coarse_tuning_stages import HypersurfaceBuilder, DoubleDotFinderViaSobol, BaseSeparationOptimisation
from pipelines.detuning_line_stages import DetuningLineDetermination, DetuningLineMsmt
from pipelines.psb_stages import PSBViaWideScan, HighResPSBClassifier, DanonGapCheck, ReCenteringStage, \
    PSBWideScanPostOptimisation
from pipelines.readout_optim import ReadoutOptim
from pipelines.spin_physics_stages import ReadoutViaBvsDetuning, RabiAndEDSR, EDSRCheck, Rabi, EDSRline, \
    RabiOscillations

from qcodes import (
    ManualParameter,
    Measurement,
    experiments,
    initialise_database,
    initialise_or_create_database_at,
    load_by_guid,
    load_by_run_spec,
    load_experiment,
    load_last_experiment,
    load_or_create_experiment,
    new_experiment,
)
from qcodes.dataset import plot_dataset
from qcodes.utils.dataset.doNd import do1d, do2d
from qcodes_addons.doNdAWG import do1dAWG, do2dAWG, init_Rabi
from utils import (
    draw_box,
    draw_boxes_and_preds,
    get_current_timestamp,
    read_row_from_file,
    timestamp_files,
)

from helper_functions.central_database_connector import JsonDBConnector
from helper_functions.data_access_layer import DataAccess
from helper_functions.documentation_generator import MarkdownFile
from helper_functions.file_system_tools import create_folder_structure  # , DataSaver

# %%
# set up experiment name and data saving etc.

# config_path = "../data/config_files/v4_scratch_to_rabi.toml"
# config_path = "../data/config_files/v8_scratch_to_rabi_gate_set_B.toml"
config_path = "../data/config_files/v9_scratch_to_rabi_gate_set_C.toml"

configs = toml.load(config_path)
experiment_name = configs['experiment_name']

create_folder_structure(experiment_name)
with open(
    "../data/experiments/" + experiment_name + "/documentation/configs.toml", "w"
) as f:
    toml.dump(configs, f)
data_access_layer = DataAccess(experiment_name)

logging.basicConfig(
    filename="../data/experiments/" + experiment_name + "/log_files/logging.log",
    format="%(asctime)s %(pathname)s line-no:%(lineno)d %(funcName)s %(levelname)s:%(message)s",
    level=logging.INFO,
)

data_access_layer.create_or_update_file(f"Starting")


# establish communication
# from experiment_control import communication_initialisation
from experiment_control.init_basel import initialise_experiment

gate_set = configs['gate_set_name']
station, *_ = initialise_experiment(gate_set=gate_set)

qcodes_parameters = {}


hypersurface_builder = HypersurfaceBuilder(station,
    experiment_name,
    qcodes_parameters,
    configs=configs["HypersurfaceBuilder"],
    data_access_layer=data_access_layer,
)

base_separation_optim = BaseSeparationOptimisation(station,
    experiment_name,
    qcodes_parameters,
    configs=configs["BaseSeparationOptimisation"],
    data_access_layer=data_access_layer,
)

double_dot_finder = DoubleDotFinderViaSobol(station,
    experiment_name,
    qcodes_parameters,
    configs=configs["DoubleDotFinder"],
    data_access_layer=data_access_layer,
)

root_stage = RootStageFromScratch(
    experiment_name, qcodes_parameters, configs["RootStage"], data_access_layer
)
terminal_stage = TerminalStage(experiment_name, {}, data_access_layer)


psb_via_wide_scan = PSBWideScanPostOptimisation(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["PSBViaWideScan"],
    data_access_layer=data_access_layer,
)

psb_high_res = HighResPSBClassifier(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["HighResPSBClassifier"],
    data_access_layer=data_access_layer,
)

#
# det_line_determination = DetuningLineDetermination(
#     station,
#     experiment_name,
#     qcodes_parameters,
#     configs=configs["DetuningLineDetermination"],
#     data_access_layer=data_access_layer,
# )
#
# det_line_scan = DetuningLineMsmt(
#     station,
#     experiment_name,
#     qcodes_parameters,
#     configs=configs["DetuningLineMsmt"],
#     data_access_layer=data_access_layer,
# )
#
# readout_b_det = ReadoutViaBvsDetuning(
#     station,
#     experiment_name,
#     qcodes_parameters,
#     configs=configs["ReadoutViaBvsDetuning"],
#     data_access_layer=data_access_layer,
# )

#
# rabi_and_edsr = RabiAndEDSR(
#     station,
#     experiment_name,
#     qcodes_parameters,
#     configs=configs["RabiAndEDSR"],
#     data_access_layer=data_access_layer,
# )


danon_check = DanonGapCheck(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["DanonGapCheck"],
    data_access_layer=data_access_layer,
)

edsr_check = EDSRCheck(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["EDSRCheck"],
    data_access_layer=data_access_layer,
)

recentering = ReCenteringStage(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["ReCenteringStage"],
    data_access_layer=data_access_layer,
)

rabi = Rabi(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["Rabi"],
    data_access_layer=data_access_layer,
)

edsr_line = EDSRline(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["EDSRline"],
    data_access_layer=data_access_layer,
)

rabi_oscillations = RabiOscillations(
    station,
    experiment_name,
    qcodes_parameters,
    configs=configs["RabiOscillations"],
    data_access_layer=data_access_layer,
)


terminal_node = TerminalStage(experiment_name, {}, data_access_layer)

qcodes_parameters = ["V_LP", "V_RP", "I_SD", "X", "Y", "R", "Phi"]
param_name_to_compat_name = {
    "V_LP": "VLP",
    "V_RP": "VRP",
    "I_SD": "ISD",
    "X": "LIX",
    "Y": "LIY",
    "R": "LIR",
    "Phi": "LIPhi",
}

readout_optim = ReadoutOptim(
    station,
    experiment_name,
    {
        param_name_to_compat_name.get(param_name, param_name): station[param_name]
        for param_name in qcodes_parameters
    },
    configs=configs["ReadoutOptim"],
    data_access_layer=data_access_layer,
)

readout_optim.name='ROOpt'



# rabi_and_edsr.child_stages = [terminal_node]
# rabi_and_edsr.parent_stages = [edsr_check]
#
#
# edsr_check.child_stages = [rabi_and_edsr]



rabi_oscillations.child_stages = [terminal_node]
rabi_oscillations.parent_stages = [rabi]


rabi.child_stages = [rabi_oscillations]
rabi.parent_stages = [edsr_line]

edsr_line.child_stages = [rabi]
edsr_line.parent_stages = [edsr_check]

edsr_check.child_stages = [edsr_line]
edsr_check.parent_stages = [readout_optim]

readout_optim.child_stages = [edsr_check]
readout_optim.parent_stages = [danon_check]
#
# det_line_scan.child_stages = [readout_optim]
# det_line_scan.parent_stages = [det_line_determination]
#
# det_line_determination.child_stages = [det_line_scan]
# det_line_determination.parent_stages = [danon_check]

danon_check.child_stages = [readout_optim]
danon_check.parent_stages = [psb_high_res]

psb_high_res.child_stages = [danon_check]
psb_high_res.parent_stages = [recentering]

recentering.child_stages = [psb_high_res]
recentering.parent_stages = [psb_via_wide_scan]

psb_via_wide_scan.child_stages = [recentering]
psb_via_wide_scan.parent_stages = [base_separation_optim]

base_separation_optim.child_stages = [psb_via_wide_scan]
base_separation_optim.parent_stages = [double_dot_finder]

double_dot_finder.child_stages = [base_separation_optim]
double_dot_finder.parent_stages = [hypersurface_builder]

hypersurface_builder.child_stages = [double_dot_finder]
hypersurface_builder.parent_stages = [root_stage]

root_stage.child_stages = [hypersurface_builder]


print("init done")

root_stage.kick_off()

# root_stage.kick_off()
