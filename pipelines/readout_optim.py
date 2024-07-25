from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import xarray as xr
from bias_triangle_detection.bayesian_optimization.bayesian_optimizer import (
    EvaluationResult,
    run_optimization,
)
from bias_triangle_detection.bayesian_optimization.parameters import (
    get_readout_optim_params,
    get_tangent_space_params,
)
from bias_triangle_detection.bayesian_optimization.setup import setup
from bias_triangle_detection.coord_change.rectangle_region import images_to_rectangle
from bias_triangle_detection.scores import ScoreType
from bias_triangle_detection.scores.edsr_peak import prominence_score
from matplotlib import pyplot as plt
from qcodes.dataset import load_by_guid

from pipelines.base_stages import Candidate
from pipelines.detuning_line_stages import DetuningLineDetermination
from pipelines.measure_and_score.score_optim import EDSRPeakScanner, Score
from pipelines.utils import plot_lockin_line_qcodes_with_overview, get_compensated_gates


class ReadoutOptim(DetuningLineDetermination):
    def get_more_candidates(self):
        # no notion of more candidates here
        # maybe this could do another optimization round
        return []

    def determine_candidates(self) -> List[Candidate]:
        self.on_analysis_start()
        unpulsed_ds = load_by_guid(self.current_candidate.info["unpulsed_msmt_id"])
        pulsed_ds = load_by_guid(
            self.current_candidate.data_identifiers["with_pulsing_measurement"]
        )
        res = self.configs["segmentation_upscaling_res"]
        relative_min_area = self.configs["relative_min_area"]
        thr_method = self.configs["thr_method"]
        invert_current = self.configs["invert_current"][self.bias_direction]
        compensated_gates_list = get_compensated_gates(self.station)
        compensated_gates_list = [comp_gate.name for comp_gate in compensated_gates_list]
        # prior_dir = 'right'
        # for comp_gate in compensated_gates_list:
        #     print(f'comp_gate.name {comp_gate.name}')
        #     if 'V_RP' == comp_gate.name:
        #         prior_dir = 'up'
        #         print('setting prior_dir to up')

        (centroid, (w, h), angle), detection_ds, fig = images_to_rectangle(
            unpulsed_ds,
            pulsed_ds,
            res,
            self.bias_direction,
            relative_min_area=relative_min_area,
            thr_method=thr_method,
            invert=invert_current,
            compensated_gates_list=compensated_gates_list
        )
        message = f"Rectangle detected, guid : {detection_ds.guid}"
        filename = f"rectangle_detection_{detection_ds.guid}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        box_data = {'centroid': centroid,
                    'w': w,
                    'h': h,
                    'angle': angle}
        self.data_access_layer.save_data({'readout_optim_box': box_data})
        self.data_access_layer.plot_readout_box(pulsed_ds.to_xarray_dataset(), box_data)
        # #####################
        # box_data = self.data_access_layer.load_data('manual_readout_optim_box')
        # self.data_access_layer.plot_readout_box(pulsed_ds.to_xarray_dataset(), box_data)
        # #####################

        rectangle_parameters = get_tangent_space_params(w, h, num_dimensions=2)
        pulse_params = get_readout_optim_params(self.configs["optim_pulse_params"])
        client = setup()
        seeding = self.configs["seeding"]
        optimizer = client.create_task(
            title="Readout Optim",
            parameters=pulse_params
            + rectangle_parameters,  # parameters are independent of surface for now
            initial_configurations=seeding,
        )
        magnetic_field_scan_config = self.configs["magnetic_field_scan_config"]
        magnetic_field_scan_config["scan_kwargs"][
            "measurement_name"
        ] = f"readout_optim_{optimizer.id}"
        experiment_config = {
            "bias_direction": self.bias_direction,
            "burst_time_ns": 1,
            "freq_vs": 1e6,
            "V_RP": 0.0,
            "V_LP": 0,
            "init": True,
            "magnetic_field": 0,
            "lockin_tc": self.configs["lockin_tc"],
        }
        experiment_config["magnetic_field"] = magnetic_field_scan_config["domain"][
            "field_setpoint"
        ]["start"]
        edsr_peak_scanner = EDSRPeakScanner(
            self.station,
            magnetic_field_scan_config,
            experiment_config=experiment_config,
            safe_mode=False,
        )
        orig_coords = ["V_RP", "V_LP"]
        target_coords = [param.name for param in rectangle_parameters]
        edsr_peak_scanner.set_transform(centroid, angle, orig_coords, target_coords)
        score_name = self.configs["score_config"]["name"]
        score_kwargs = self.configs["score_config"]["kwargs"]
        scorer = getattr(ScoreType, score_name.upper()).value(**score_kwargs)

        metadata = {"task_id": optimizer.id}
        point_to_score = Score(scorer, edsr_peak_scanner, metadata=metadata)

        results: List[EvaluationResult] = run_optimization(
            optimizer, point_to_score, **self.configs["optimization_kwargs"],
            data_access_layer=self.data_access_layer
        )
        number_of_candidates = self.configs["number_of_candidates"]
        candidates_infos = []
        prom_score = prominence_score()
        overview_data = load_by_guid(self.current_candidate.info['overview_data_id']).to_xarray_dataset()
        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        scores = [result.score for result in results]
        results = np.array(results)[np.argsort(scores)][::-1].tolist()
        n_candidates_found = 0
        for i, result in enumerate(results):
            info = asdict(result)
            info["config"] = edsr_peak_scanner.optim_point_to_voltage_point(
                result.config
            )
            candidate_info = deepcopy(self.current_candidate.info)
            candidate_info.update(info)
            candidate_info['pulsed_msmt_id'] = self.current_candidate.data_identifiers["with_pulsing_measurement"]
            message = f'candidate #{i}, score: {result.score}\n' \
                      f'config: {result.config}, metadata: {result.metadata}'
            trace = load_by_guid(result.metadata["guid"]).to_xarray_dataset()
            lp = candidate_info['config']['V_LP']
            rp = candidate_info['config']['V_RP']
            fig = plot_lockin_line_qcodes_with_overview(trace, [rp, lp], pulsed_ds.to_xarray_dataset(), overview_data, location, sidelength )
            fig_name = f'readout_optim_candidate_{i}'
            self.data_access_layer.create_with_figure(message, fig, fig_name)
            plt.close()
            has_multiple_peaks, peak_locs_magnet = get_peaks(result.metadata["guid"], prom_score)
            if has_multiple_peaks:
                self.data_access_layer.create_or_update_file(f'has multiple or no peaks: {peak_locs_magnet}, disregard candidate')
                continue
            self.data_access_layer.create_or_update_file(f'has one or two peaks {peak_locs_magnet}, build candidate with first one')
            candidate_info["magnetic_field"] = float(peak_locs_magnet[0])
            candidates_infos.append(candidate_info)
            n_candidates_found += 1
            if n_candidates_found == number_of_candidates:
                break
        candidates = self.on_analysis_end(candidates_infos)
        return candidates


def get_peaks(guid: str, prom_scorer: prominence_score) -> (bool, np.array):
    """Load dataset and check if it has multiple peaks along the magnetic field."""
    # load dataset
    trace = load_by_guid(guid).to_xarray_dataset()
    # get peaks
    lockin_variables = list(trace.data_vars)
    magnetic_field_name = next(iter(trace.dims))
    locs, _ = xr.apply_ufunc(
        prom_scorer._call,
        trace[lockin_variables].to_array(),
        input_core_dims=[[magnetic_field_name, "variable"]],
        output_core_dims=[[magnetic_field_name], []],
        vectorize=True,
    )
    # has multiple peaks
    key = list(dict(trace.dims).keys())[0]
    idx = np.argwhere(locs.to_numpy() == 1)
    locs_magnet = trace[key].to_numpy()[idx]
    return (locs.sum().item() > 2 or locs.sum().item() == 0), locs_magnet


if __name__ == "__main__":
    import logging
    import pickle
    from pathlib import Path

    import toml

    from experiment_control.dummy_init import initialise_dummy_experiment
    from helper_functions.data_access_layer import DataAccess
    from helper_functions.file_system_tools import (
        create_folder_structure,  # , DataSaver
    )
    from pipelines.base_stages import TerminalStage
    from pipelines.detuning_line_stages import DetuningLineDetermination
    from pipelines.readout_optim import ReadoutOptim

    experiment_name = "20230711_testing_readout_optim"
    comment = "testing readout optim node"
    data_folder = Path(__file__).parent.parent / "data"
    config_path = data_folder / "config_files" / "v3.toml"

    create_folder_structure(experiment_name)
    configs = toml.load(config_path)
    experiment_folder = data_folder / "experiments" / experiment_name
    experiment_config_path = experiment_folder / "documentation/configs.toml"
    # create experiment_config dir if it not exists
    if not experiment_config_path.parent.exists():
        experiment_config_path.parent.mkdir(parents=True)

    with open(
        experiment_config_path,
        "w",
    ) as f:
        toml.dump(configs, f)
    data_access_layer = DataAccess(experiment_name)
    log_path = experiment_folder / "log_files/logging.log"
    # create log dir if it not exists
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True)
    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s %(pathname)s line-no:%(lineno)d %(funcName)s %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    data_access_layer.create_or_update_file(f"Starting")

    # establish communication
    # from experiment_control import communication_initialisation
    from experiment_control.init_basel import initialise_experiment

    test_mode = False
    if test_mode:
        station = initialise_dummy_experiment("data/GeSiNW_Qubit_VTI01_Jonas_2.db")
    else:
        station, *_ = initialise_experiment()

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

    terminal_node = TerminalStage(experiment_name, {}, data_access_layer)

    readout_optim.child_stages = [terminal_node]

    print("init done")

    candidate = []
    path_pickle = (
        data_folder
        # / "experiments/20230630_full_test/node_states/root_0/Root_0/WideScan_10/Cntr_0/HRPSBClf_0/Danon_0/DetLineDet.pkl"
        / "experiments/20230711_test_new_code/node_states/root_0/Root_0/WideScan_0/Cntr_0/HRPSBClf_0/Danon_0/DetLineDet.pkl"
    )
    with open(path_pickle, "rb") as f:
        candidate = pickle.load(f)
    readout_optim.investigate(candidate)
