import heapq
import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List

import loguru
import numpy as np
from bias_triangle_detection.bayesian_optimization.bayesian_optimizer import (
    EvaluationResult,
    run_optimization,
)
from bias_triangle_detection.bayesian_optimization.parameters import (
    get_tangent_space_params,
)
from bias_triangle_detection.bayesian_optimization.setup import setup
from bias_triangle_detection.coord_change import NamedFunc
from bias_triangle_detection.coord_change.tangent_to_surface import (
    get_model,
    train_model,
)
from bias_triangle_detection.cotunnelling_detection import get_cotunneling
from bias_triangle_detection.scores import ScoreType
from matplotlib import pyplot as plt
from qcodes import load_by_guid, load_by_id
from qcodes.dataset import do0d, do1d, do2d
from qcodes_addons.Parameterhelp import GateParameter, VirtualGateParameter
from scipy.stats import qmc
from tqdm import tqdm

from experiment_control.init_basel import detuning
from helper_functions.data_access_layer import DataAccess
from pipelines.base_stages import BaseStage, Candidate
from pipelines.characterisation_stages import NoiseFloorExtraction
from pipelines.coarse_tuning.score_optim import (
    CoulombPeakScanner,
    PointInDomainAndTargetSpace,
    get_domain_bounds,
    get_target_bounds,
)
from pipelines.coarse_tuning.test_score_optim import Scorer
from pipelines.coarse_tuning.test_score_optimisation_reinstation import (
    run_base_separation_optimisation,
)
from pipelines.utils import (
    plot_qcodes_data,
    plot_qcodes_line_data,
    ramp_magnet_before_msmt,
    timestamp_files,
)
from search_algorithms.binary_search.binary_search import BinarySearch
from signal_processing.coulomb_peak_classifier.coulomb_peak_classifier import (
    rfc_peak_check,
)
from signal_processing.high_res_double_dot_clf.dd_clf import HighResDDClassifier
from bias_triangle_detection.switches.fastai_model_wrapper import ArrayGetter, LabelGetter, FastAISwitchModel


class CornerFinding(BaseStage):
    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Corner"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.magnetic_field = float(configs["magnetic_field"])

        self.barriers = ["VL", "VM", "VR"]
        self.plungers = ["VLP", "VRP"]
        self.gate_names = ["VL", "VLP", "VM", "VRP", "VR"]
        self.gates = [
            self.station["V_L"],
            self.station["V_LP"],
            self.station["V_M"],
            self.station["V_RP"],
            self.station["V_R"],
        ]

        def jump(value_to_jump_to):
            logging.debug(f"jump, received: {value_to_jump_to}")
            # assert isinstance(value_to_jump_to, tuple)
            if isinstance(value_to_jump_to, tuple):
                gate_name = value_to_jump_to[1]
                values = value_to_jump_to[0]
            else:
                gate_name = None
                values = value_to_jump_to

            if gate_name is None:
                for gate, value in zip(self.gates, values):
                    gate(value)
            else:
                idx = self.gate_names.index(gate_name)
                self.gates[idx](values)

        self.jump = jump

        def measure():
            return self.station["I_SD"]()

        self.measure = measure
        threshold = float(configs["threshold"])
        lower_is_inside = bool(configs["lower_is_inside"])
        binary_search_stopping_distance = float(
            configs["binary_search_stopping_distance"]
        )

        self.binary_search = BinarySearch(
            self.jump,
            self.measure,
            threshold,
            lower_is_inside,
            binary_search_stopping_distance,
        )

        self.data_xarray = None

        self.bias_direction = None

        #######################

        self.first_origin = configs["first_origin"]
        self.plunger_boundaries = configs["plunger_boundaries"]
        self.threshold = configs["threshold"]
        self.stepsize = configs["stepsize"]
        self.bounds = configs["bounds"]
        self.n_iter = configs["n_iter"]
        self.lower_is_inside = configs["lower_is_inside"]
        self.binary_search_stopping_distance = configs[
            "binary_search_stopping_distance"
        ]
        self.investigation_range = configs["investigation_range"]
        self.plotting = configs["plotting"]

        # logging.info(("Barrier gates: ", self.barriers))
        # logging.info(("plunger gates: ", self.plungers))

        self.pinch_offs_found = {}
        self.investigation_results = {}
        self.dd_found_history = {}
        self.xyz_iters = {}  # Stores the moving origin of the barriers
        self.plunger_record = {}
        self.fixed_gates_record = {}

        self.pinch_off_finder = self.binary_search
        # self.data_saver = data_saver
        path_to_nn = configs["path_to_nn"]
        self.double_dot_classifier = HighResDDClassifier(path_to_nn=path_to_nn)
        self.coulomb_peak_classifier = rfc_peak_check
        self.data_2d_xarray = None
        self.data_1d_xarray = None

    def scan_barrier_and_find_pinch_off(
        self, barrier_init: float, barrier_index: int
    ) -> float:
        """Find the pinch off between "barrier_init" and its bound
        given in "bounds[barrier_index]"

            Args
                barrier_init: Bound on the given barrier
                barrier_index: Index of the barrier in "bounds"/"barriers"
            Returns
                (float): Found pinch off
        """
        barrier_name = self.barriers[barrier_index]
        bounds = self.bounds[barrier_index]

        logging.debug(("Scanning barrier, received: ", barrier_init, barrier_name))

        self.pinch_off_finder.set_gate_to_check(barrier_name)
        pinch_off = self.pinch_off_finder.perform_search(barrier_init, bounds)

        self.jump(value_to_jump_to=(barrier_init, barrier_name))

        return pinch_off

    def investigate(self, candidate: Candidate):
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.current_candidate.data_identifiers["2d_ids"] = []
            self.current_candidate.data_identifiers["1d_ids"] = []

            dd_found, all_gates, _, _ = self.run_corner_finding()

            fig = self.plot()
            message = (
                f"corner finding done, found double dot: {dd_found}"
                f", all_gates {all_gates}"
            )
            filename = f"corner_finding"
            self.data_access_layer.create_with_figure(message, fig, filename)

            self.current_candidate.info["dd_found"] = dd_found
            self.current_candidate.info["all_gates"] = all_gates
            self.save_state()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")
            dd_found = self.current_candidate.info["dd_found"]
            all_gates = self.current_candidate.info["all_gates"]

        if dd_found:
            candidate_info = self.current_candidate.info
            # candidates = self.build_candidates([candidate_info])
            candidate_info_list = [candidate_info]
        else:
            candidate_info_list = []

        # self.current_candidate.resulting_candidates = candidates
        candidates = self.on_analysis_end(candidate_info_list)

        # hand off to next stage

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def run_corner_finding(self) -> (list, dict):
        """Perform the corner finding search.

        Returns
            (list): Whether double dots were found and at which iteration
            (dict): Contains information about gate voltages
        """

        # for repetition in range(self.n_reps):
        """
        For each repetition, the randomised gates are set to
        new values and the origin is moved back to the original
        setting.
        """
        repetition = 0
        self.repetition = repetition
        logging.info(("Repetition: ", repetition))

        self.prepare_measurement()

        first_origin = np.array(self.first_origin)
        rand_nums = np.random.rand(len(first_origin))
        origin_settings = (
            first_origin[:, 0] + (first_origin[:, 1] - first_origin[:, 0]) * rand_nums
        )
        logging.info(("Origin set to: ", origin_settings))

        self.xyz_iters[repetition] = [origin_settings]
        # (
        #     plunger_settings,
        #     fixed_gates_settings,
        # ) = self.find_initial_gate_voltage_values()
        plunger_settings = [0, 0]

        # initialise arrays for data collection
        self.pinch_offs_found[repetition] = []
        self.investigation_results[repetition] = []
        self.dd_found_history[repetition] = []
        self.plunger_record[repetition] = plunger_settings
        # self.fixed_gates_record[repetition] = fixed_gates_settings

        self.data_access_layer.save_data({"plunger_record": self.plunger_record})
        self.data_access_layer.save_data(
            {
                "fixed_gates_record": self.fixed_gates_record,
            }
        )

        for iteration in range(self.n_iter):
            logging.info(("Repetition, iteration: ", repetition, iteration))
            all_gates = self.update_gates_array(
                plunger_settings, origin_settings  # , fixed_gates_settings
            )
            logging.info(("Setting all gates to ", all_gates))
            self.jump(value_to_jump_to=all_gates)

            new_origin = self.find_pinch_offs_and_new_origin(*origin_settings)
            origin_settings = new_origin

            all_gates = self.update_gates_array(
                plunger_settings,
                origin_settings,
                # fixed_gates_settings,
            )

            logging.info(("Setting all gates to ", all_gates))
            self.jump(value_to_jump_to=all_gates)

            # investigation stage
            results_of_this_investigation = self.investigate_region(all_gates)

            self.investigation_results[repetition].append(results_of_this_investigation)
            dd_found = results_of_this_investigation[0]
            self.dd_found_history[repetition].append(dd_found)
            logging.info(f"dd_found: {dd_found}")

            if dd_found:
                print("*" * 50)
                print("*" * 50)
                print("Success! Double Dot found!")
                print("Gate voltages:", results_of_this_investigation[1])
                print("*" * 50)
                print("*" * 50)
                break

            # self.plot()
            self.data_access_layer.save_data(
                {
                    "pinch_offs_found": self.pinch_offs_found,
                    "xyz_iters": self.xyz_iters,
                    "dd_found_history": self.dd_found_history,
                    "investigation_results": self.investigation_results,
                },
            )

        return dd_found, all_gates, self.dd_found_history, self.investigation_results

    def find_pinch_offs_and_new_origin(
        self, x_init: float, y_init: float, z_init: float
    ) -> list:
        """Find pinch off along each barrier gate axis and move origin.

        Args:
            x_init (float): Initial barrier value
            y_init (float): Initial barrier value
            z_init (float): Initial barrier value
        Returns:
            (list): New origin

        """
        x_poff = self.scan_barrier_and_find_pinch_off(
            barrier_init=x_init,
            barrier_index=0,
        )
        logging.info(("x_poff", x_poff))

        y_poff = self.scan_barrier_and_find_pinch_off(
            barrier_init=y_init,
            barrier_index=1,
        )
        logging.info(("y_poff", y_poff))

        z_poff = self.scan_barrier_and_find_pinch_off(
            barrier_init=z_init,
            barrier_index=2,
        )
        logging.info(("z_poff", z_poff))

        logging.info(("pinch offs", x_poff, y_poff, z_poff))
        self.pinch_offs_found[self.repetition].append((x_poff, y_poff, z_poff))

        # find the next point as a proportion of distance to hypersurface
        xyz_poffs = np.array([x_poff, y_poff, z_poff])  # Pinch off coordinates
        xyz_init = np.array([x_init, y_init, z_init])  # Current origin
        delta_vec = xyz_poffs - xyz_init
        xyz_next = xyz_init + self.stepsize * delta_vec

        logging.debug(("xyz_poffs", xyz_poffs))
        logging.debug(("xyz_init", xyz_init))
        logging.debug(("delta_vec", delta_vec))
        logging.debug(("xyz_next", xyz_next))

        x_next, y_next, z_next = xyz_next  # Next origin
        self.xyz_iters[self.repetition].append((x_next, y_next, z_next))

        logging.info(("Origin set to", xyz_next))
        return xyz_next

    def update_gates_array(
        self,
        plunger_settings: list,
        xyz: list,  # fixed_gates_settings: list
    ) -> list:
        """Create array of gate voltages for jump function

        Helper function that goes through all the gate names and gives back a single array
        that can be fed into the 'jump' function

            Args
                plunger_settings (list of floats): current voltage settings of plungers
                xyz (list of floats): current voltage settings of barriers
                fixed_gates_settings (list of floats): current voltage settings of fixed gates
            Returns
                (array): voltage of all gates as an array
        """
        all_gates = [0 for _ in self.gate_names]
        for i, n in enumerate(self.plungers):
            all_gates[self.gate_names.index(n)] = plunger_settings[i]
        for i, n in enumerate(self.barriers):
            all_gates[self.gate_names.index(n)] = xyz[i]
        # for i, n in enumerate(self.fixed_gates):
        #     all_gates[self.gate_names.index(n)] = fixed_gates_settings[i]
        return all_gates

    def plot(self):
        """Plot data.

        Two things can be plotted: The history of the origin,
        or the history of the origin and the history of the pinch offs
        found along each barrier axis.
        """

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_title("Hypersurface and origins")
        data = np.array(self.pinch_offs_found[self.repetition])
        xdata = data[:, 0]
        ydata = data[:, 1]
        zdata = data[:, 2]
        data_origin = np.array(self.xyz_iters[self.repetition])
        xdata_origin = data_origin[:, 0][:-1]
        ydata_origin = data_origin[:, 1][:-1]
        zdata_origin = data_origin[:, 2][:-1]
        ax.set_xlabel("x (" + str(self.barriers[0]) + ")")
        ax.set_ylabel("y(" + str(self.barriers[1]) + ")")
        ax.set_zlabel("z(" + str(self.barriers[2]) + ")")
        ax.scatter3D(xdata, ydata_origin, zdata_origin)
        ax.scatter3D(xdata_origin, ydata, zdata_origin)
        ax.scatter3D(xdata_origin, ydata_origin, zdata)
        data = np.array(self.xyz_iters[self.repetition])
        xdata = data[:, 0]
        ydata = data[:, 1]
        zdata = data[:, 2]
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="hot")
        # plt.show()
        return fig

    def prepare_measurement(self):
        self.station['V_SD'](self.configs["bias_voltage"])
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        self.station.awg.stop()

    def do_coulomb_peak_measurement(self, rp_center, lp_center):
        window_rp = self.configs["1d_scan"]["window_right_plunger"]
        window_lp = self.configs["1d_scan"]["window_left_plunger"]
        # n_px_rp = self.configs["1d_scan"]["n_px_rp"]
        n_px = self.configs["1d_scan"]["n_px"]
        # wait_time_slow_axis = self.configs["1d_scan"]["wait_time_slow_axis"]
        wait_time = self.configs["1d_scan"]["wait_time"]
        rp_start = rp_center - window_rp / 2
        rp_end = rp_center + window_rp / 2
        lp_start = lp_center - window_lp / 2
        lp_end = lp_center + window_lp / 2

        current_candidate_name = self.state.candidate_names[
            self.state.current_candidate_idx
        ]
        params_det = detuning(lp_start, rp_start, lp_end, rp_end)
        detuning_line = VirtualGateParameter(
            name="det_" + current_candidate_name,
            params=(self.station["V_LP"], self.station["V_RP"]),
            set_scaling=(1, params_det[0]),
            offsets=(0, params_det[1]),
        )

        data_handle = do1d(
            detuning_line,
            lp_start,
            lp_end,
            n_px,
            wait_time,
            self.station["I_SD"],
            show_progress=True,
        )
        data_handle = data_handle[0]

        msmt_id = data_handle.run_id
        self.current_candidate.data_identifiers["1d_measurement"] = msmt_id
        self.data_1d_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_line_data(self.data_1d_xarray)

        message = f"2d scan done, high magnetic field, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.current_candidate.data_identifiers["1d_ids"].append(msmt_id)

    def do_2d_scan(self, rp_center, lp_center):
        window_rp = self.configs["2d_scan"]["window_right_plunger"]
        window_lp = self.configs["2d_scan"]["window_left_plunger"]
        n_px_rp = self.configs["2d_scan"]["n_px_rp"]
        n_px_lp = self.configs["2d_scan"]["n_px_lp"]
        wait_time_slow_axis = self.configs["2d_scan"]["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["2d_scan"]["wait_time_fast_axis"]
        rp_start = rp_center - window_rp / 2
        rp_end = rp_center + window_rp / 2
        lp_start = lp_center - window_lp / 2
        lp_end = lp_center + window_lp / 2

        data_handle = do2d(
            self.station["V_RP"],
            rp_start,
            rp_end,
            n_px_rp,
            wait_time_slow_axis,
            self.station["V_LP"],
            lp_start,
            lp_end,
            n_px_lp,
            wait_time_fast_axis,
            self.station["I_SD"],
            show_progress=True,
        )
        data_handle = data_handle[0]

        msmt_id = data_handle.run_id
        self.current_candidate.data_identifiers["2d_measurement"] = msmt_id
        self.data_2d_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_data(self.data_2d_xarray)

        message = f"2d scan done, high magnetic field, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.current_candidate.data_identifiers["2d_ids"].append(msmt_id)

    def check_for_coulomb_peaks(self, rp_center, lp_center):
        self.do_coulomb_peak_measurement(rp_center, lp_center)
        prediction = self.coulomb_peak_classifier(
            self.data_1d_xarray["I_SD"].to_numpy()
        )
        return prediction

    def check_for_double_dot(self, rp_center, lp_center):
        self.do_2d_scan(rp_center, lp_center)
        prediction = self.double_dot_classifier.predict(
            self.data_2d_xarray["I_SD"].to_numpy()
        )
        return prediction

    def investigate_region(self, rp_center, lp_center):
        cp_found = self.check_for_coulomb_peaks(rp_center, lp_center)
        dd_found = False
        if cp_found:
            dd_found = self.check_for_double_dot(rp_center, lp_center)

        return dd_found, cp_found


class ConductingDetectorThreshold(object):
    def __init__(self, th_high):
        self.th_high = th_high

    def __call__(self, trace, invert_current=True):
        if invert_current:
            high = -trace > -self.th_high
            # print(f'high {high}')
            # print(f'-trace {-trace}')
            # print(f'-self.th_high {-self.th_high}')
        else:
            high = trace > self.th_high

        idxs = np.where(high)[0]
        if np.size(idxs) == 0:
            return -1
        return idxs[0]  # return the first index satistying the condition


class PinchoffDetectorThreshold(object):
    def __init__(self, th_low):
        self.th_low = th_low

    def __call__(self, trace, reverse_direction=False, invert_current=True):
        if reverse_direction is True:  # reverse search
            trace = trace[::-1]
        if invert_current:
            low = -trace < -self.th_low
            # print(f'low {low}')
            # print(f'-trace {-trace}')
            # print(f'-self.th_low {-self.th_low}')
        else:
            low = trace < self.th_low
        # change_points is 1 when ~low -> low, or low -> ~low (~: not)
        # change_points = np.logical_xor(low[:-1], np.logical_not(low[1:]))
        change_points = np.logical_xor(low[:-1], low[1:])
        change_points = np.concatenate(
            ([True], change_points), axis=0
        )  # change point is true for the first point

        possible_points = np.logical_and(low, change_points)

        # high_prev indicates whether it was above the thresholdshold at least once
        # (not high enough -> low is not pinchoff)
        # high_prev = np.concatenate( ([False], (trace>self.th_high)[:-1]), axis=0 )
        # for i in range(1,trace.size):
        #    high_prev[i] = np.logical_or(high_prev[i-1], high_prev[i])
        # possible_points = np.logical_and(possible_points,high_prev)

        # keep_low indicates whether it keeps low afterwards
        keep_low = np.concatenate((low[1:], [True]), axis=0)
        for i in range(trace.size - 2, -1, -1):
            keep_low[i] = np.logical_and(keep_low[i], keep_low[i + 1])

        idxs = np.where(np.logical_and(possible_points, keep_low))[0]
        if np.size(idxs) == 0:
            return -1

        idx = idxs[0]
        if reverse_direction is True:  # reverse search
            idx = np.size(trace) - idx - 1
        return idx


def L2_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))


def check_inside_boundary(voltages, lb, ub):
    if np.any(voltages < lb) or np.any(voltages > ub):
        return False
    return True


class LineSearch(object):
    def __init__(
        self,
        station,
        jump,
        measure,
        real_lb,
        real_ub,
        detector_pinchoff,
        detector_conducting,
        d_r=0.005,
        len_after_pinchoff=0.250,
        max_dist=2,
        logging=False,
        bias_low=0,
        bias_high=0.005,
    ):
        self.station = station
        self.jump = jump  # connection
        self.measure = measure
        self.lb = np.array(real_lb)  # lower bound
        self.ub = np.array(real_ub)  # upper bound
        self.d_r = d_r  # step size (default)
        self.detector_pinchoff = detector_pinchoff
        self.detector_conducting = detector_conducting
        self.len_after_pinchoff = len_after_pinchoff
        self.max_dist = max_dist
        self.bias_low = bias_low
        self.bias_high = bias_high

    def search_line(self, voltages, unit_vector, ignore_idxs):
        """
        Args:
            detector: function that returns a scalar integer of a detected point
            ignore_idxs: list of integer indice, search won't stop if pinchoff idx is in the list
        Returns:
            voltages_all: list of 1D array (num of points for a line trace, num of gates)
            measurement_all: list of scalar vals (num of points for a line trace)
            found_idx: scalar, -1 indicates nothing detected across the whole line, or the voltages are out of bound
                                0 indicates detected at the starting point

        """
        voltages_from = voltages.copy()
        voltages_all_outwards = list()
        measurement_all_outwards = list()
        found_idx_outwards = -1

        self.station["V_SD"](self.bias_low)
        print(f"setting voltage to {self.bias_low}")
        # There are two lines that terminate the while loop
        # 1. checking the voltages is still in the boundary, and distance < max_dist, break if not
        # 2. break if poff detected, but check some signal afterwards within 'len_after'
        first_iter = True
        while (
            check_inside_boundary(voltages, self.lb, self.ub)
            and L2_norm(voltages - voltages_from) < self.max_dist
        ):
            # print(f'jumping to {voltages}')
            self.jump(voltages)  # set voltages for measurement
            if first_iter:
                first_iter = False

            current = self.measure()  # current measurement
            voltages_all_outwards.append(voltages.copy())  # location of the measurement
            measurement_all_outwards.append(current)
            # print(f'current {current}')
            found_idx_outwards = self.detector_pinchoff(
                np.array(measurement_all_outwards)
            )
            if (
                found_idx_outwards not in ignore_idxs
                and L2_norm(
                    voltages_all_outwards[-1]
                    - voltages_all_outwards[found_idx_outwards]
                )
                >= self.len_after_pinchoff
            ):
                # pinchoff found, not 0, and measured enough length after the found pinchoff
                break
            voltages = voltages + self.d_r * unit_vector  # go futher to theta direction

        self.station["V_SD"](self.bias_high)
        print(f"setting voltage to {self.bias_high}")

        voltages_from = voltages.copy()
        voltages_all = list()
        measurement_all = list()
        found_idx = -1

        first_iter = True
        voltages = voltages - self.d_r * unit_vector
        while (
            check_inside_boundary(voltages, self.lb, self.ub)
            and L2_norm(voltages - voltages_from) < self.max_dist
        ):
            # print(f'jumping to {voltages}')
            self.jump(voltages)  # set voltages for measurement
            if first_iter:
                first_iter = False

            current = self.measure()  # current measurement
            # print(f'current {current}')
            voltages_all.append(voltages.copy())  # location of the measurement
            measurement_all.append(current)
            found_idx = self.detector_conducting(np.array(measurement_all))
            if found_idx not in ignore_idxs:
                break
            voltages = voltages - self.d_r * unit_vector  # go futher to theta direction

        self.station["V_SD"](self.bias_low)
        print(f"setting voltage to {self.bias_low}")

        return (
            voltages_all,
            measurement_all,
            found_idx,
            voltages_all_outwards,
            measurement_all_outwards,
            found_idx_outwards,
        )


def sample_unit_vectors(n, dims, use_sobol=True, upper_bounds=(1,1,1)):
    """Sample n unit vectors uniformly distributed over a hypersphere surface in given dimensions.

    Oversamples the corner.

    Args:
        n (int): The number of vectors to generate.
        dims (int): The number of dimensions in the hyperspace.

    Returns:
        np.ndarray: An array of shape (n, dims) where each row is a unit vector.
    """
    if use_sobol:
        # Generate Sobol sequence
        sobol_engine = qmc.Sobol(d=dims, scramble=True)
        vectors = sobol_engine.random(n=n)
        vectors = qmc.scale(vectors, (0,0,0), upper_bounds)
    else:
        vectors = np.random.normal(
            size=(n, dims)
        )  # Generate Gaussian random numbers for each dimension
    vectors = np.abs(vectors)  # Make all dimensions positive
    return (
        vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    )  # Normalize to obtain unit vectors


class HypersurfaceBuilder(BaseStage):
    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "HSBldr" # "HypersurfaceBuilder"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.bias_low = configs["bias_low"]
        self.bias_high = configs["bias_high"]

        self.upper_bounds = configs['upper_bounds']
        self.barriers = ["VL", "VM", "VR"]
        self.plungers = ["VLP", "VRP"]
        self.gate_names = ["VL", "VLP", "VM", "VRP", "VR"]
        self.gates = [
            self.station["V_L"],
            self.station["V_LP"],
            self.station["V_M"],
            self.station["V_RP"],
            self.station["V_R"],
        ]
        self.detector_pinchoff = None
        self.detector_conducting = None
        self.line_search = None

    def _init_poff_detector(self):
        print(f"Initialising pinch off detector")
        threshold_as_multiple_of_noise_low = self.configs[
            "threshold_as_multiple_of_noise_low"
        ]
        threshold_as_multiple_of_noise_high = self.configs[
            "threshold_as_multiple_of_noise_high"
        ]

        noise_floor_extraction = NoiseFloorExtraction(self.station)
        n_noise_floor = self.configs["n_noise_floor"]
        current_at_origin, current_at_pinchoff = noise_floor_extraction.get_noise_floor(
            n_noise_floor, [1.5, 1.5, 1.5], self.bias_low, self.bias_high
        )
        print(f"current_at_origin {current_at_origin}")
        print(f"current_at_pinchoff {current_at_pinchoff[0]}")
        print(f"current_at_pinchoff {current_at_pinchoff[1]}")
        threshold_low = (
            np.array(current_at_pinchoff[0]["current"])
            - np.array(current_at_pinchoff[0]["current_std"])
            * threshold_as_multiple_of_noise_low
        )
        threshold_high = (
            np.array(current_at_pinchoff[1]["current"])
            - np.array(current_at_pinchoff[0]["current_std"])
            * threshold_as_multiple_of_noise_high
        )
        print(f"thresholds {threshold_low, threshold_high}")
        self.detector_pinchoff = PinchoffDetectorThreshold(
            threshold_low
        )  # pichoff detector
        self.detector_conducting = ConductingDetectorThreshold(
            threshold_high
        )  # conducting detector
        lower_bounds = self.configs["lower_bounds"]
        upper_bounds = self.configs["upper_bounds"]
        d_r = self.configs["d_r"]
        len_after_pinchoff = self.configs["len_after_pinchoff"]
        max_dist = self.configs["max_dist"]
        self.line_search = LineSearch(
            self.station,
            self.jump,
            self.measure,
            lower_bounds,
            upper_bounds,
            self.detector_pinchoff,
            detector_conducting=self.detector_conducting,
            d_r=d_r,
            len_after_pinchoff=len_after_pinchoff,
            max_dist=max_dist,
            bias_low=self.bias_low,
            bias_high=self.bias_high,
        )

    def prepare_measurement(self):
        self.station['V_SD'](self.bias_low)
        self.station.awg.stop()
        left_plunger_value = self.configs['plunger_location'][0] if ('plunger_location' in self.configs) else 0
        right_plunger_value = self.configs['plunger_location'][1] if ('plunger_location' in self.configs) else 0
        self.station['V_LP'](left_plunger_value)
        self.station['V_RP'](right_plunger_value)

        if self.line_search is None:
            self._init_poff_detector()

    def jump(self, value_to_jump_to):
        logging.debug(f"jump, received: {value_to_jump_to}")
        # print(f"jump, received: {value_to_jump_to}")
        # assert isinstance(value_to_jump_to, tuple)
        if isinstance(value_to_jump_to, tuple):
            gate_name = value_to_jump_to[1]
            values = value_to_jump_to[0]
        else:
            gate_name = None
            values = value_to_jump_to

        if gate_name is None:
            if len(values) == 3:  # Assuming only barriers
                # usrinput= input(f'Setting barriers {barriers},\n to values {values}, y or n?')
                # if usrinput == 'y':
                for barrier_name, value in zip(self.barriers, values):
                    idx = self.gate_names.index(barrier_name)
                    self.gates[idx](value)
                else:
                    pass
            else:
                usrinput = input(
                    f"Setting gates {self.gate_names},\n to values {values}, y or n?"
                )
                if usrinput == "y":
                    for gate, value in zip(self.gates, values):
                        gate(value)
                else:
                    pass
        else:
            idx = self.gate_names.index(gate_name)
            usrinput = input(
                f"Setting gate_name {gate_name}, idx {idx} to value {values}, y or n?"
            )
            if usrinput == "y":
                self.gates[idx](values)

    def measure(self):
        return self.station[
            "I_SD"
        ]()  # do0d(ISD)[0].to_xarray_dataset()['I_SD'].to_numpy()

    def get_pinch_off_locations(self, origin, n=100):
        unit_vectors = sample_unit_vectors(n, len(self.barriers), upper_bounds=self.upper_bounds)
        print(f"unit_vectors {unit_vectors}")
        # unit_vectors = [unit_vectors[0], 0 , unit_vectors[1], 0, unit_vectors[2]]
        # offset_for_pinchoff =configs["offset_for_pinchoff"]
        # print(f'unit_vectors {unit_vectors}')
        voltages_all_collected = []
        measurement_all_collected = []
        found_idx_collected = []
        voltages_all_collected_out = []
        measurement_all_collected_out = []
        found_idx_collected_out = []
        for i, unit_vector in tqdm(enumerate(unit_vectors)):
            # print(f'unit_vector {unit_vector}')
            (
                voltages_all,
                measurement_all,
                found_idx,
                v_outs,
                msmt_out,
                idx_out,
            ) = self.line_search.search_line(
                voltages=origin, unit_vector=unit_vector, ignore_idxs=[-1]
            )
            voltages_all_collected.append(voltages_all)
            measurement_all_collected.append(measurement_all)
            found_idx_collected.append(found_idx)
            voltages_all_collected_out.append(v_outs)
            measurement_all_collected_out.append(msmt_out)
            found_idx_collected_out.append(idx_out)

            if i % self.configs["plot_every"] == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.scatter(
                    np.linalg.norm(np.array(voltages_all), axis=1), measurement_all
                )
                ax1.set_xlabel("distance from origin")
                ax1.set_title("inwards")
                ax2.scatter(np.linalg.norm(np.array(v_outs), axis=1), msmt_out)
                ax2.set_xlabel("distance from origin")
                ax2.set_title("outwards")
                message = f"inward poff {voltages_all[found_idx]}, outward poff {v_outs[idx_out]}"
                figname = "pinchoff_trace"
                self.data_access_layer.create_with_figure(message, fig, figname)
                plt.close()

        return (
            voltages_all_collected,
            measurement_all_collected,
            found_idx_collected,
            voltages_all_collected_out,
            measurement_all_collected_out,
            found_idx_collected_out,
        )

    def perform_measurements(self):
        self.on_measurement_start()
        n_rays = self.configs["number_of_rays"]

        origin = np.array([0, 0, 0])
        (
            voltages_all_collected,
            measurement_all_collected,
            found_idx_collected,
            v_outs,
            msmt_out,
            idx_out,
        ) = self.get_pinch_off_locations(origin, n_rays)
        pinch_off_locations = []
        for volts, idx in zip(voltages_all_collected, found_idx_collected):
            pinch_off_locations.append(volts[idx])
        pinch_off_locations = np.array(pinch_off_locations)

        unit_vector = np.array([1, 0, 0])
        (
            voltages_all_first_axis,
            measurement_all_first_axis,
            found_idx_first_axis,
            v_outs1,
            msmt_out1,
            idx_out1,
        ) = self.line_search.search_line(
            voltages=origin, unit_vector=unit_vector, ignore_idxs=[-1]
        )

        unit_vector = np.array([0, 1, 0])
        (
            voltages_all_second_axis,
            measurement_all_second_axis,
            found_idx_second_axis,
            v_outs2,
            msmt_out2,
            idx_out2,
        ) = self.line_search.search_line(
            voltages=origin, unit_vector=unit_vector, ignore_idxs=[-1]
        )

        unit_vector = np.array([0, 0, 1])
        (
            voltages_all_third_axis,
            measurement_all_third_axis,
            found_idx_third_axis,
            v_outs3,
            msmt_out3,
            idx_out3,
        ) = self.line_search.search_line(
            voltages=origin, unit_vector=unit_vector, ignore_idxs=[-1]
        )

        time_now = timestamp_files()
        major_axis_voltages_all_collected_name = time_now + "_voltages_major_axis"
        self.data_access_layer.save_data(
            {
                major_axis_voltages_all_collected_name: {
                    'first axis':{
                        "inward": voltages_all_first_axis,
                        "outward": v_outs1,
                },
                    'second axis':{
                        "inward": voltages_all_second_axis,
                        "outward": v_outs2,
                },
                    'third axis':{
                        "inward": voltages_all_third_axis,
                        "outward": v_outs3,
                }
                }
            }
        )
        major_axis_measurement_all_collected_name = time_now + "_measurement_major_axis"
        self.data_access_layer.save_data(
            {
                major_axis_measurement_all_collected_name: {
                    'first axis': {
                    "inward": measurement_all_first_axis,
                    "outward": msmt_out1
                    },
                    'second axis': {
                        "inward": measurement_all_second_axis,
                        "outward": msmt_out2
                    },
                    'third axis': {
                    "inward": measurement_all_third_axis,
                    "outward": msmt_out3
                    },
                }
            }
        )

        poff_1 = voltages_all_first_axis[found_idx_first_axis]
        poff_2 = voltages_all_second_axis[found_idx_second_axis]
        poff_3 = voltages_all_third_axis[found_idx_third_axis]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # Also plot the original data points for reference
        ax.scatter(
            pinch_off_locations[:, 0],
            pinch_off_locations[:, 1],
            pinch_off_locations[:, 2],
            c="r",
            s=50,
            label="random direction pinchoffs",
        )
        ax.scatter(poff_1[0], poff_1[1], poff_1[2], c="b", s=50)
        ax.scatter(poff_2[0], poff_2[1], poff_2[2], c="b", s=50)
        ax.scatter(
            poff_3[0], poff_3[1], poff_3[2], c="b", s=50, label="major axes pinchoff"
        )
        ax.set_xlabel(self.barriers[0])
        ax.set_ylabel(self.barriers[1])
        ax.set_zlabel(self.barriers[2])
        plt.legend()
        message = (
            f"Pinch off hypersurface sampled at with bias voltage {self.bias_high} and"
            f"number of rays {n_rays}\n"
            f"Major axis pinch pinch offs: {poff_1, poff_2, poff_3}"
        )
        figname = "hypersurface"

        self.data_access_layer.create_with_figure(message, fig, figname)
        plt.close()

        major_axis_poff = [poff_1[0], poff_2[1], poff_3[2]]

        major_axis_poff_name = time_now + "_major_axis_poff"
        self.data_access_layer.save_data({major_axis_poff_name: major_axis_poff})
        pinch_off_locations_name = time_now + "_pinch_off_locations"
        self.data_access_layer.save_data(
            {pinch_off_locations_name: pinch_off_locations}
        )

        voltages_all_collected_name = time_now + "_voltages_all_collected"
        self.data_access_layer.save_data(
            {
                voltages_all_collected_name: {
                    "inward": voltages_all_collected,
                    "outward": v_outs,
                }
            }
        )
        measurement_all_collected_name = time_now + "_measurement_all_collected"
        self.data_access_layer.save_data(
            {
                measurement_all_collected_name: {
                    "inward": measurement_all_collected,
                    "outward": msmt_out,
                }
            }
        )
        found_idx_collected_name = time_now + "_found_idx_collected"
        self.data_access_layer.save_data(
            {
                found_idx_collected_name: {
                    "inward": found_idx_collected,
                    "outward": idx_out,
                }
            }
        )

        # misusing this, as those are not qcodes ids
        self.current_candidate.data_identifiers[
            "major_axis_poff_name"
        ] = major_axis_poff_name
        self.current_candidate.data_identifiers["major_axis_poff"] = major_axis_poff
        self.current_candidate.data_identifiers[
            "pinch_off_locations_name"
        ] = pinch_off_locations_name
        self.on_measurement_end()

    def investigate(self, candidate: Candidate):
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.prepare_measurement()
            self.perform_measurements()
            self.current_candidate.data_taken = True
            self.save_state()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")

        self.on_analysis_start()
        candidate_info = {
            "major_axis_poff_name": self.current_candidate.data_identifiers[
                "major_axis_poff_name"
            ],
            "pinch_off_locations_name": self.current_candidate.data_identifiers[
                "pinch_off_locations_name"
            ],
            "major_axis_poff": self.current_candidate.data_identifiers[
                "major_axis_poff"
            ],
            "bias_direction": self.bias_direction,
        }

        candidates = self.on_analysis_end([candidate_info])
        # candidates = self.build_candidates([candidate_info])
        #
        # self.current_candidate.resulting_candidates = candidates
        # hand off to next stage

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)


class DoubleDotFinderViaSobol(BaseStage):
    def __init__(
            self,
            station,
            experiment_name: str,
            qcodes_parameters: Dict,
            configs: dict,
            data_access_layer: DataAccess,
    ):
        name = "DDFndr"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.magnetic_field = float(configs["magnetic_field"])

        self.barriers = ["VL", "VM", "VR"]
        self.plungers = ["VLP", "VRP"]
        self.gate_names = ["VL", "VLP", "VM", "VRP", "VR"]
        self.gates = [
            self.station["V_L"],
            self.station["V_LP"],
            self.station["V_M"],
            self.station["V_RP"],
            self.station["V_R"],
        ]
        path_to_nn = self.configs["path_to_nn"]
        self.double_dot_classifier = HighResDDClassifier(path_to_nn=path_to_nn)
        self.coulomb_peak_classifier = rfc_peak_check
        self.xyz_iters = []
        self.dd_found_history = []
        self.investigation_results = []
        self.data_2d_xarray = None
        self.data_1d_xarray = None

    def prepare_measurement(self):
        self.station['V_SD'](self.configs["bias_voltage"])
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        self.station.awg.stop()

    def jump(self, value_to_jump_to):
        logging.debug(f"jump, received: {value_to_jump_to}")
        if isinstance(value_to_jump_to, tuple):
            gate_name = value_to_jump_to[1]
            values = value_to_jump_to[0]
        else:
            gate_name = None
            values = value_to_jump_to

        if gate_name is None:
            if len(values) == 3:  # Assuming only barriers
                for barrier_name, value in zip(self.barriers, values):
                    idx = self.gate_names.index(barrier_name)
                    self.gates[idx](value)
                else:
                    pass
            else:
                usrinput = input(
                    f"Setting gates {self.gate_names},\n to values {values}, y or n?"
                )
                if usrinput == "y":
                    for gate, value in zip(self.gates, values):
                        gate(value)
                else:
                    pass
        else:
            idx = self.gate_names.index(gate_name)
            usrinput = input(
                f"Setting gate_name {gate_name}, idx {idx} to value {values}, y or n?"
            )
            if usrinput == "y":
                self.gates[idx](values)

    def perform_measurements(self):
        self.on_measurement_start()
        normalise_pinchoffs = self.configs['normalise_pinchoffs'] if 'normalise_pinchoffs' in self.configs else True

        origin = self.configs['origin']
        major_axis_poff_name = self.current_candidate.info["major_axis_poff_name"]
        pinch_off_locations_name = self.current_candidate.info[
            "pinch_off_locations_name"
        ]
        major_axis_poff = np.array(self.data_access_layer.load_data(major_axis_poff_name))
        pinch_off_locations = np.array(self.data_access_layer.load_data(pinch_off_locations_name))
        if normalise_pinchoffs:
            pinch_off_locations_corrected = pinch_off_locations - origin
            major_axis_poff_corrected = np.array(major_axis_poff) - origin
            major_axis_poff_translated = np.array(major_axis_poff) - origin
            pinch_off_locations_corrected /= major_axis_poff_corrected
            major_axis_poff_corrected /= major_axis_poff_corrected
        else:
            pinch_off_locations_corrected = pinch_off_locations
            major_axis_poff_corrected = np.array(major_axis_poff)

        import torch
        from scipy.optimize import minimize

        # Conversion functions
        def cart_to_spherical(x, y, z):
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
            phi = np.arctan2(y, x)
            return r, theta, phi

        def spherical_to_cart(r, theta, phi):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return x, y, z

        # Prepare the data
        X = pinch_off_locations_corrected[:, :2]
        y = pinch_off_locations_corrected[:, 2]
        r, theta, phi = cart_to_spherical(X[:, 0], X[:, 1], y)
        train_x = torch.from_numpy(np.column_stack([theta, phi]))
        train_y = torch.from_numpy(r)

        # Create and train the model
        model = get_model(train_x, train_y)
        model = train_model(model, train_x, train_y).pred_mean

        # def objective_function(x, vector, model):
        #     r_pred = model(torch.from_numpy(x[np.newaxis, :])).detach().numpy()
        #     x_pred, y_pred, z_pred = spherical_to_cart(r_pred, x[0], x[1])
        #     return np.linalg.norm(np.array([x_pred, y_pred, z_pred]) - vector)

        def objective_function(x, vector, model):
            r_pred = (
                model(torch.from_numpy(x[np.newaxis, :])).detach().numpy()[0]
            )  # Making sure we get a single value
            x_pred, y_pred, z_pred = spherical_to_cart(r_pred, x[0], x[1])
            return np.linalg.norm(np.array([x_pred, y_pred, z_pred]) - vector)

        major_axis_poff_spherical = cart_to_spherical(*major_axis_poff_corrected)
        x0 = np.mean(np.column_stack([theta, phi]), axis=0)

        result = minimize(objective_function, x0, args=(major_axis_poff_corrected, model))
        r_projected = (
            model(torch.from_numpy(result.x[np.newaxis, :])).detach().numpy()[0]
        )
        projected_point = spherical_to_cart(r_projected, *result.x)
        if normalise_pinchoffs:
            projected_point_retransformed = projected_point * major_axis_poff_translated + origin
        else:
            projected_point_retransformed = projected_point

        # theta_range = np.linspace(theta.min(), theta.max(), num=50)
        # phi_range = np.linspace(phi.min(), phi.max(), num=50)

        theta_range = np.linspace(0, np.pi / 2, num=50)
        phi_range = np.linspace(0, np.pi / 2, num=50)

        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
        X_grid = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])
        # r_grid_pred = model(torch.from_numpy(X_grid)).detach().numpy()
        # x_grid_pred, y_grid_pred, z_grid_pred = spherical_to_cart(r_grid_pred, theta_grid, phi_grid)
        r_grid_pred = (
            model(torch.from_numpy(X_grid)).detach().numpy().reshape(theta_grid.shape)
        )
        x_grid_pred, y_grid_pred, z_grid_pred = spherical_to_cart(
            r_grid_pred, theta_grid, phi_grid
        )

        # Visualization
        perspectives = [(30, 30), (30, 90), (30, 120), (30, 150)]
        fig = plt.figure(figsize=(4, 15))
        for i, (elev, azim) in enumerate(perspectives):
            ax = fig.add_subplot(4, 1, i + 1, projection="3d")
            ax.plot_surface(
                x_grid_pred,
                y_grid_pred,
                z_grid_pred,
                cmap="icefire",
                alpha=0.7,
                edgecolor="none",
                zorder=1,
            )
            ax.scatter(X[:, 0], X[:, 1], y, c="r", s=50, label="measurements")
            ax.scatter(
                *projected_point, c="b", s=100, label="projected point", zorder=0
            )
            ax.scatter(
                *major_axis_poff_corrected, marker="x", c="g", s=50, label="Major axis pinch off"
            )
            ax.set_xlabel(self.barriers[0]+', normalised')
            ax.set_ylabel(self.barriers[1]+', normalised')
            ax.set_zlabel(self.barriers[2]+', normalised')
            ax.view_init(elev, azim)
            plt.legend()
            plt.title(f"Perspective {i}: elev={elev}, azim={azim}")

        message = f"hypersurface with projected point: {projected_point, projected_point_retransformed}"
        figname = "hypersurface_with_projection"
        self.data_access_layer.create_with_figure(message, fig, figname)
        plt.close()

        self.current_candidate.data_identifiers["1d_ids"] = []
        self.current_candidate.data_identifiers["2d_ids"] = []
        # start either from major_axis_poff towards projected_point, or start at projected point and move towards origin
        barrier_settings = np.array(projected_point_retransformed)

        max_distance_between_locations = self.configs['max_distance_between_locations']
        for n_samples in range(12):
            sampler = qmc.Sobol(d=3, scramble=False)
            sample = sampler.random_base2(m=n_samples)

            sample_scaled = qmc.scale(sample, barrier_settings, major_axis_poff)
            all_distances = np.linalg.norm(sample_scaled - sample_scaled[:, None], axis=2)
            min_distance_to_neighbour = np.min(all_distances + np.eye(len(sample_scaled)) * 1000, axis=1)
            if np.max(min_distance_to_neighbour) < max_distance_between_locations:
                break
        sample_scaled = np.vstack(([barrier_settings], sample_scaled))
        if normalise_pinchoffs:
            x_grid_pred_corr = x_grid_pred * major_axis_poff_translated[0] + origin[0]
            y_grid_pred_corr = y_grid_pred * major_axis_poff_translated[1] + origin[1]
            z_grid_pred_corr = z_grid_pred * major_axis_poff_translated[2] + origin[2]
        else:
            x_grid_pred_corr = x_grid_pred
            y_grid_pred_corr = y_grid_pred
            z_grid_pred_corr = z_grid_pred
        # Visualization
        perspectives = [(30, 30), (30, 90), (30, 120), (30, 150)]
        fig = plt.figure(figsize=(4, 15))
        for i, (elev, azim) in enumerate(perspectives):
            ax = fig.add_subplot(4, 1, i + 1, projection="3d")
            ax.plot_surface(
                x_grid_pred_corr,
                y_grid_pred_corr,
                z_grid_pred_corr,
                cmap="icefire",
                alpha=0.7,
                edgecolor="none",
                zorder=1,
            )
            ax.scatter(pinch_off_locations[:, 0], pinch_off_locations[:, 1], pinch_off_locations[:, 2], c="r", s=50,
                       label="measurements")
            ax.scatter(sample_scaled[:, 0], sample_scaled[:, 1], sample_scaled[:, 2],
                       c="black", s=50, label="proposed locations")
            ax.scatter(
                *projected_point_retransformed, c="b", s=100, label="projected point", zorder=0
            )
            ax.scatter(
                *major_axis_poff, marker="x", c="g", s=50, label="Major axis pinch off"
            )
            ax.set_xlabel(self.barriers[0])
            ax.set_ylabel(self.barriers[1])
            ax.set_zlabel(self.barriers[2])
            ax.view_init(elev, azim)
            plt.legend()
            plt.title(f"Perspective {i}: elev={elev}, azim={azim}")

        message = f"hypersurface with proposed locations, number: {len(sample_scaled)}," \
                  f" distance between locations: {np.max(min_distance_to_neighbour)}"
        figname = "hypersurface_with_projection_and_proposed_locations"
        self.data_access_layer.create_with_figure(message, fig, figname)
        plt.close()
        distance_to_center = np.linalg.norm(sample_scaled - projected_point_retransformed, axis=1)
        idx_sorted = np.argsort(distance_to_center)
        sample_scaled = sample_scaled[idx_sorted]

        for iteration, barrier_settings in enumerate(sample_scaled):
            logging.info(("iteration: ", iteration))
            self.iteration = iteration
            self.barrier_settings = barrier_settings

            logging.info(("Setting barriers to ", barrier_settings))
            self.jump(value_to_jump_to=barrier_settings)

            # investigation stage
            left_plunger_value = self.configs['plunger_location'][0] if ('plunger_location' in self.configs) else 0
            right_plunger_value = self.configs['plunger_location'][1] if ('plunger_location' in self.configs) else 0
            results_of_this_investigation = self.investigate_region(
                lp_center=left_plunger_value, rp_center=right_plunger_value
            )

            self.investigation_results.append(results_of_this_investigation)
            dd_found = results_of_this_investigation[0]
            self.dd_found_history.append(dd_found)
            logging.info(f"dd_found: {dd_found}")

            if dd_found:
                print("*" * 50)
                print("*" * 50)
                print("Success! Double Dot found!")
                print("Gate voltages:", barrier_settings)
                print("*" * 50)
                print("*" * 50)
                break

            # self.plot()
            self.data_access_layer.save_data(
                {
                    "xyz_iters": self.xyz_iters,
                    "dd_found_history": self.dd_found_history,
                    "investigation_results": self.investigation_results,
                },
            )

        self.on_measurement_end()
        return dd_found, barrier_settings

    def do_coulomb_peak_measurement(self, rp_center, lp_center):
        window_rp = self.configs["1d_scan"]["window_right_plunger"]
        window_lp = self.configs["1d_scan"]["window_left_plunger"]
        # n_px_rp = self.configs["1d_scan"]["n_px_rp"]
        n_px = self.configs["1d_scan"]["n_px"]
        # wait_time_slow_axis = self.configs["1d_scan"]["wait_time_slow_axis"]
        wait_time = self.configs["1d_scan"]["wait_time"]
        rp_start = rp_center - window_rp / 2
        rp_end = rp_center + window_rp / 2
        lp_start = lp_center - window_lp / 2
        lp_end = lp_center + window_lp / 2

        current_candidate_name = self.state.candidate_names[
            self.state.current_candidate_idx
        ]
        params_det = detuning(lp_start, rp_start, lp_end, rp_end)
        detuning_line = VirtualGateParameter(
            name=("det_" + current_candidate_name + "_" + str(self.iteration)).replace(
                "-", "_"
            ),
            params=(self.station["V_LP"], self.station["V_RP"]),
            set_scaling=(1, params_det[0]),
            offsets=(0, params_det[1]),
        )

        data_handle = do1d(
            detuning_line,
            lp_start,
            lp_end,
            n_px,
            wait_time,
            self.station["I_SD"],
            show_progress=True,
            measurement_name="coulomb_peaks_detection",
        )
        data_handle = data_handle[0]

        msmt_id = data_handle.run_id
        self.current_candidate.data_identifiers["1d_measurement"] = msmt_id
        self.data_1d_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_line_data(self.data_1d_xarray)

        message = f"coloumb peak measurement done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.current_candidate.data_identifiers["1d_ids"].append(msmt_id)

    def do_2d_scan(self, rp_center, lp_center):
        window_rp = self.configs["2d_scan"]["window_right_plunger"]
        window_lp = self.configs["2d_scan"]["window_left_plunger"]
        n_px_rp = self.configs["2d_scan"]["n_px_rp"]
        n_px_lp = self.configs["2d_scan"]["n_px_lp"]
        wait_time_slow_axis = self.configs["2d_scan"]["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["2d_scan"]["wait_time_fast_axis"]
        rp_start = rp_center - window_rp / 2
        rp_end = rp_center + window_rp / 2
        lp_start = lp_center - window_lp / 2
        lp_end = lp_center + window_lp / 2

        data_handle = do2d(
            self.station["V_RP"],
            rp_start,
            rp_end,
            n_px_rp,
            wait_time_slow_axis,
            self.station["V_LP"],
            lp_start,
            lp_end,
            n_px_lp,
            wait_time_fast_axis,
            self.station["I_SD"],
            show_progress=True,
            measurement_name="double_dot_detection",
        )
        data_handle = data_handle[0]

        msmt_id = data_handle.run_id
        self.current_candidate.data_identifiers["2d_measurement"] = msmt_id
        self.data_2d_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_data(self.data_2d_xarray)

        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.current_candidate.data_identifiers["2d_ids"].append(msmt_id)

    def check_for_coulomb_peaks(self, rp_center, lp_center):
        self.do_coulomb_peak_measurement(rp_center, lp_center)
        prediction = self.coulomb_peak_classifier(
            -self.data_1d_xarray["I_SD"].to_numpy()
        )
        self.data_access_layer.create_or_update_file(
            f"Coloumb peak check result: {prediction} at "
            f"barrier voltages {self.barrier_settings}"
        )
        return prediction

    def check_for_double_dot(self, rp_center, lp_center):
        self.do_2d_scan(rp_center, lp_center)
        prediction = self.double_dot_classifier.predict(
            -self.data_2d_xarray["I_SD"].to_numpy()
        )
        self.data_access_layer.create_or_update_file(
            f"Double dot check result: {prediction} at "
            f"barrier voltages {self.barrier_settings}"
        )
        return prediction

    def investigate_region(self, rp_center, lp_center):
        cp_found = self.check_for_coulomb_peaks(rp_center, lp_center)
        dd_found = False
        if cp_found:
            dd_found = self.check_for_double_dot(rp_center, lp_center)

        return dd_found, cp_found

    def investigate(self, candidate: Candidate):
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.bias_voltage = self.configs["bias_voltage"]
            self.prepare_measurement()
            dd_found, barrier_setting = self.perform_measurements()
            self.current_candidate.data_identifiers["dd_found"] = dd_found
            self.current_candidate.data_identifiers["barrier_setting"] = barrier_setting
            self.current_candidate.data_taken = True
            self.save_state()
        else:
            print(f"Data already taken for {self.name} node")
            dd_found = self.current_candidate.data_identifiers["dd_found"]

        self.on_analysis_start()
        if dd_found:
            candidate_info = {
                "major_axis_poff_name": self.current_candidate.info[
                    "major_axis_poff_name"
                ],
                "upper_bound": self.current_candidate.info["major_axis_poff"],
                "pinch_off_locations_name": self.current_candidate.info[
                    "pinch_off_locations_name"
                ],
                "lower_bound": self.current_candidate.data_identifiers[
                    "barrier_setting"
                ],
                "bias_direction": "positive_bias",
            }
            candidate_info_list = [candidate_info]
            # candidates = self.build_candidates([candidate_info])
        else:
            # candidates = []
            candidate_info_list = []
        candidates = self.on_analysis_end(candidate_info_list)
        # self.current_candidate.resulting_candidates = candidates
        # self.current_candidate.analysis_done = True
        # self.save_state()
        # hand off to next stage

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)


class DoubleDotFinder(BaseStage):
    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "DDFndr"#"DoubleDotFinder"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.magnetic_field = float(configs["magnetic_field"])

        self.barriers = ["VL", "VM", "VR"]
        self.plungers = ["VLP", "VRP"]
        self.gate_names = ["VL", "VLP", "VM", "VRP", "VR"]
        self.gates = [
            self.station["V_L"],
            self.station["V_LP"],
            self.station["V_M"],
            self.station["V_RP"],
            self.station["V_R"],
        ]
        path_to_nn = self.configs["path_to_nn"]
        self.double_dot_classifier = HighResDDClassifier(path_to_nn=path_to_nn)
        self.coulomb_peak_classifier = rfc_peak_check
        self.xyz_iters = []
        self.dd_found_history = []
        self.investigation_results = []
        self.data_2d_xarray = None
        self.data_1d_xarray = None

    def prepare_measurement(self):
        self.station['V_SD'](self.configs["bias_voltage"])
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        self.station.awg.stop()

    def jump(self, value_to_jump_to):
        logging.debug(f"jump, received: {value_to_jump_to}")
        if isinstance(value_to_jump_to, tuple):
            gate_name = value_to_jump_to[1]
            values = value_to_jump_to[0]
        else:
            gate_name = None
            values = value_to_jump_to

        if gate_name is None:
            if len(values) == 3:  # Assuming only barriers
                for barrier_name, value in zip(self.barriers, values):
                    idx = self.gate_names.index(barrier_name)
                    self.gates[idx](value)
                else:
                    pass
            else:
                usrinput = input(
                    f"Setting gates {self.gate_names},\n to values {values}, y or n?"
                )
                if usrinput == "y":
                    for gate, value in zip(self.gates, values):
                        gate(value)
                else:
                    pass
        else:
            idx = self.gate_names.index(gate_name)
            usrinput = input(
                f"Setting gate_name {gate_name}, idx {idx} to value {values}, y or n?"
            )
            if usrinput == "y":
                self.gates[idx](values)

    def perform_measurements(self):
        self.on_measurement_start()
        major_axis_poff_name = self.current_candidate.info["major_axis_poff_name"]
        pinch_off_locations_name = self.current_candidate.info[
            "pinch_off_locations_name"
        ]
        major_axis_poff = self.data_access_layer.load_data(major_axis_poff_name)
        pinch_off_locations = self.data_access_layer.load_data(pinch_off_locations_name)

        import torch
        from scipy.optimize import minimize

        # Conversion functions
        def cart_to_spherical(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)
            phi = np.arctan2(y, x)
            return r, theta, phi

        def spherical_to_cart(r, theta, phi):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return x, y, z

        # Prepare the data
        X = pinch_off_locations[:, :2]
        y = pinch_off_locations[:, 2]
        r, theta, phi = cart_to_spherical(X[:, 0], X[:, 1], y)
        train_x = torch.from_numpy(np.column_stack([theta, phi]))
        train_y = torch.from_numpy(r)

        # Create and train the model
        model = get_model(train_x, train_y)
        model = train_model(model, train_x, train_y).pred_mean

        # def objective_function(x, vector, model):
        #     r_pred = model(torch.from_numpy(x[np.newaxis, :])).detach().numpy()
        #     x_pred, y_pred, z_pred = spherical_to_cart(r_pred, x[0], x[1])
        #     return np.linalg.norm(np.array([x_pred, y_pred, z_pred]) - vector)

        def objective_function(x, vector, model):
            r_pred = (
                model(torch.from_numpy(x[np.newaxis, :])).detach().numpy()[0]
            )  # Making sure we get a single value
            x_pred, y_pred, z_pred = spherical_to_cart(r_pred, x[0], x[1])
            return np.linalg.norm(np.array([x_pred, y_pred, z_pred]) - vector)

        major_axis_poff_spherical = cart_to_spherical(*major_axis_poff)
        x0 = np.mean(np.column_stack([theta, phi]), axis=0)

        result = minimize(objective_function, x0, args=(major_axis_poff, model))
        r_projected = (
            model(torch.from_numpy(result.x[np.newaxis, :])).detach().numpy()[0]
        )
        projected_point = spherical_to_cart(r_projected, *result.x)

        # theta_range = np.linspace(theta.min(), theta.max(), num=50)
        # phi_range = np.linspace(phi.min(), phi.max(), num=50)

        theta_range = np.linspace(0, np.pi / 2, num=50)
        phi_range = np.linspace(0, np.pi / 2, num=50)

        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
        X_grid = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])
        # r_grid_pred = model(torch.from_numpy(X_grid)).detach().numpy()
        # x_grid_pred, y_grid_pred, z_grid_pred = spherical_to_cart(r_grid_pred, theta_grid, phi_grid)
        r_grid_pred = (
            model(torch.from_numpy(X_grid)).detach().numpy().reshape(theta_grid.shape)
        )
        x_grid_pred, y_grid_pred, z_grid_pred = spherical_to_cart(
            r_grid_pred, theta_grid, phi_grid
        )

        # Visualization
        perspectives = [(30, 30), (30, 90), (30, 120), (30, 150)]
        fig = plt.figure(figsize=(4, 15))
        for i, (elev, azim) in enumerate(perspectives):
            ax = fig.add_subplot(4, 1, i + 1, projection="3d")
            ax.plot_surface(
                x_grid_pred,
                y_grid_pred,
                z_grid_pred,
                cmap="icefire",
                alpha=0.7,
                edgecolor="none",
                zorder=1,
            )
            ax.scatter(X[:, 0], X[:, 1], y, c="r", s=50, label="measurements")
            ax.scatter(
                *projected_point, c="b", s=100, label="projected point", zorder=0
            )
            ax.scatter(
                *major_axis_poff, marker="x", c="g", s=50, label="Major axis pinch off"
            )
            ax.set_xlabel(self.barriers[0])
            ax.set_ylabel(self.barriers[1])
            ax.set_zlabel(self.barriers[2])
            ax.view_init(elev, azim)
            plt.legend()
            plt.title(f"Perspective {i}: elev={elev}, azim={azim}")

        # # Define the perspectives to use
        # perspectives = [(30, 30), (30, 90), (30, 120), (30, 150)]
        #
        # fig = plt.figure(figsize = (4,15))
        #
        # for i, (elev, azim) in enumerate(perspectives):
        #     ax = fig.add_subplot(4,1,i+1, projection='3d')
        #     ax.plot_surface(x_grid, y_grid, z_grid_pred.reshape(x_grid.shape), cmap='icefire', alpha=0.7,
        #                     edgecolor='none', zorder=1)
        #
        #     ax.scatter(X[:, 0], X[:, 1], y, c='r', s=50, label='measurements')
        #     ax.scatter(*projected_point, c='b', s=100, label='projected point', zorder=0)
        #     ax.scatter(major_axis_poff[0], major_axis_poff[1], major_axis_poff[2], marker='x', c='g', s=50,
        #                label='Major axis pinch off')
        #
        #     ax.set_xlabel(self.barriers[0])
        #     ax.set_ylabel(self.barriers[1])
        #     ax.set_zlabel(self.barriers[2])
        #
        #     # Set the perspective
        #     ax.view_init(elev, azim)
        #
        #     plt.legend()
        #     plt.title(f'Perspective {i}: elev={elev}, azim={azim}')

        message = f"hypersurface with projected point: {projected_point}"
        figname = "hypersurface_with_projection"
        self.data_access_layer.create_with_figure(message, fig, figname)
        plt.close()

        self.current_candidate.data_identifiers["1d_ids"] = []
        self.current_candidate.data_identifiers["2d_ids"] = []
        # start either from major_axis_poff towards projected_point, or start at projected point and move towards origin
        barrier_settings = np.array(projected_point)


        # try skipping this
        dd_found=True
        """for iteration in range(self.configs["n_iter"]):
            logging.info(("iteration: ", iteration))
            self.iteration = iteration
            self.barrier_settings = barrier_settings

            logging.info(("Setting barriers to ", barrier_settings))
            self.jump(value_to_jump_to=barrier_settings)

            # investigation stage
            results_of_this_investigation = self.investigate_region(
                lp_center=0, rp_center=0
            )

            self.investigation_results.append(results_of_this_investigation)
            dd_found = results_of_this_investigation[0]
            self.dd_found_history.append(dd_found)
            logging.info(f"dd_found: {dd_found}")

            if dd_found:
                print("*" * 50)
                print("*" * 50)
                print("Success! Double Dot found!")
                print("Gate voltages:", barrier_settings)
                print("*" * 50)
                print("*" * 50)
                break

            # self.plot()
            self.data_access_layer.save_data(
                {
                    "xyz_iters": self.xyz_iters,
                    "dd_found_history": self.dd_found_history,
                    "investigation_results": self.investigation_results,
                },
            )
            if not dd_found:
                delta_vec = barrier_settings / np.linalg.norm(barrier_settings)
                barrier_settings = (
                    barrier_settings - delta_vec * self.configs["step_size"]
                )
                self.xyz_iters.append(barrier_settings)"""
        self.on_measurement_end()
        return dd_found, barrier_settings

    def do_coulomb_peak_measurement(self, rp_center, lp_center):
        window_rp = self.configs["1d_scan"]["window_right_plunger"]
        window_lp = self.configs["1d_scan"]["window_left_plunger"]
        # n_px_rp = self.configs["1d_scan"]["n_px_rp"]
        n_px = self.configs["1d_scan"]["n_px"]
        # wait_time_slow_axis = self.configs["1d_scan"]["wait_time_slow_axis"]
        wait_time = self.configs["1d_scan"]["wait_time"]
        rp_start = rp_center - window_rp / 2
        rp_end = rp_center + window_rp / 2
        lp_start = lp_center - window_lp / 2
        lp_end = lp_center + window_lp / 2

        current_candidate_name = self.state.candidate_names[
            self.state.current_candidate_idx
        ]
        params_det = detuning(lp_start, rp_start, lp_end, rp_end)
        detuning_line = VirtualGateParameter(
            name=("det_" + current_candidate_name + "_" + str(self.iteration)).replace(
                "-", "_"
            ),
            params=(self.station["V_LP"], self.station["V_RP"]),
            set_scaling=(1, params_det[0]),
            offsets=(0, params_det[1]),
        )

        data_handle = do1d(
            detuning_line,
            lp_start,
            lp_end,
            n_px,
            wait_time,
            self.station["I_SD"],
            show_progress=True,
            measurement_name="coulomb_peaks_detection",
        )
        data_handle = data_handle[0]

        msmt_id = data_handle.run_id
        self.current_candidate.data_identifiers["1d_measurement"] = msmt_id
        self.data_1d_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_line_data(self.data_1d_xarray)

        message = f"coloumb peak measurement done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.current_candidate.data_identifiers["1d_ids"].append(msmt_id)

    def do_2d_scan(self, rp_center, lp_center):
        window_rp = self.configs["2d_scan"]["window_right_plunger"]
        window_lp = self.configs["2d_scan"]["window_left_plunger"]
        n_px_rp = self.configs["2d_scan"]["n_px_rp"]
        n_px_lp = self.configs["2d_scan"]["n_px_lp"]
        wait_time_slow_axis = self.configs["2d_scan"]["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["2d_scan"]["wait_time_fast_axis"]
        rp_start = rp_center - window_rp / 2
        rp_end = rp_center + window_rp / 2
        lp_start = lp_center - window_lp / 2
        lp_end = lp_center + window_lp / 2

        data_handle = do2d(
            self.station["V_RP"],
            rp_start,
            rp_end,
            n_px_rp,
            wait_time_slow_axis,
            self.station["V_LP"],
            lp_start,
            lp_end,
            n_px_lp,
            wait_time_fast_axis,
            self.station["I_SD"],
            show_progress=True,
            measurement_name="double_dot_detection",
        )
        data_handle = data_handle[0]

        msmt_id = data_handle.run_id
        self.current_candidate.data_identifiers["2d_measurement"] = msmt_id
        self.data_2d_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_data(self.data_2d_xarray)

        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.current_candidate.data_identifiers["2d_ids"].append(msmt_id)

    def check_for_coulomb_peaks(self, rp_center, lp_center):
        self.do_coulomb_peak_measurement(rp_center, lp_center)
        prediction = self.coulomb_peak_classifier(
            -self.data_1d_xarray["I_SD"].to_numpy()
        )
        self.data_access_layer.create_or_update_file(
            f"Coloumb peak check result: {prediction} at "
            f"barrier voltages {self.barrier_settings}"
        )
        return prediction

    def check_for_double_dot(self, rp_center, lp_center):
        self.do_2d_scan(rp_center, lp_center)
        prediction = self.double_dot_classifier.predict(
            -self.data_2d_xarray["I_SD"].to_numpy()
        )
        self.data_access_layer.create_or_update_file(
            f"Double dot check result: {prediction} at "
            f"barrier voltages {self.barrier_settings}"
        )
        return prediction

    def investigate_region(self, rp_center, lp_center):
        cp_found = self.check_for_coulomb_peaks(rp_center, lp_center)
        dd_found = False
        if cp_found:
            dd_found = self.check_for_double_dot(rp_center, lp_center)

        return dd_found, cp_found

    def investigate(self, candidate: Candidate):
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.bias_voltage = self.configs["bias_voltage"]
            self.prepare_measurement()
            dd_found, barrier_setting = self.perform_measurements()
            self.current_candidate.data_identifiers["dd_found"] = dd_found
            self.current_candidate.data_identifiers["barrier_setting"] = barrier_setting
            self.current_candidate.data_taken = True
            self.save_state()
        else:
            print(f"Data already taken for {self.name} node")
            dd_found = self.current_candidate.data_identifiers["dd_found"]

        self.on_analysis_start()
        if dd_found:
            candidate_info = {
                "major_axis_poff_name": self.current_candidate.info[
                    "major_axis_poff_name"
                ],
                "upper_bound": self.current_candidate.info["major_axis_poff"],
                "pinch_off_locations_name": self.current_candidate.info[
                    "pinch_off_locations_name"
                ],
                "lower_bound": self.current_candidate.data_identifiers[
                    "barrier_setting"
                ],
                "bias_direction": "positive_bias",
            }
            candidate_info_list = [candidate_info]
            # candidates = self.build_candidates([candidate_info])
        else:
            # candidates = []
            candidate_info_list = []
        candidates = self.on_analysis_end(candidate_info_list)
        # self.current_candidate.resulting_candidates = candidates
        # self.current_candidate.analysis_done = True
        # self.save_state()
        # hand off to next stage

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)


class BaseSeparationOptimisation(BaseStage):
    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "BSO"#"BaseSepOpt"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters


    def investigate(self, candidate: Candidate):
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.bias_voltage = self.configs["bias_voltage"]
            self.perform_measurements()
            self.current_candidate.data_taken = True
            self.save_state()
        else:
            print(f"Data already taken for {self.name} node")

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            self.determine_candidates()
        candidates = self.current_candidate.resulting_candidates
        for candidate in candidates:
            for gate in ["V_L", "V_M", "V_R"]:
                self.station[gate](candidate.info["config"][gate])
            if candidate.info["bias_direction"] == "positive_bias":
                sign = 1
            else:
                sign = -1
            self.station["V_SD"](sign * self.configs["bias_voltage"])
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def determine_candidates(self):
        self.on_analysis_start()
        name_results = self.current_candidate.data_identifiers["name_results"]

        results = self.data_access_layer.load_data(name_results)

        number_of_candidates = self.configs["number_of_candidates"]
        candidates_infos = []
        number_of_candidates = np.min((number_of_candidates, len(results)))
        for i in range(number_of_candidates):
            bias_directions = self.configs["bias_directions"]
            for bias_direction in bias_directions:
                result = results[i]

                info = asdict(result)
                info["config"] = result.config
                candidate_info = deepcopy(self.current_candidate.info)
                candidate_info.update(info)
                candidate_info.update(result.metadata['scan_domain'])
                # candidate_info["left_plunger_voltage"] = self.configs[
                #     "left_plunger_voltage"
                # ]
                # candidate_info["right_plunger_voltage"] = self.configs[
                #     "right_plunger_voltage"
                # ]
                candidate_info["bias_direction"] = bias_direction
                candidates_infos.append(candidate_info)
            message = (
                f"candidate #{i}, score: {result.score}\n"
                f"considering bias directions: {bias_directions}\n"
                f"config: {result.config}, metadata: {result.metadata}"
            )
            stab_diagram = load_by_guid(result.metadata["guid"]).to_xarray_dataset()
            lp = candidate_info["V_LP"]
            rp = candidate_info["V_RP"]
            rp_start = rp['start']
            rp_stop = rp['stop']
            lp_start = lp['start']
            lp_stop = lp['stop']
            box = np.array([
                [lp_start, rp_start],
                [lp_stop, rp_start],
                [lp_stop, rp_stop],
                [lp_start, rp_stop],
                [lp_start, rp_start]
            ])
            fig = plot_qcodes_data(stab_diagram, box=box)
            fig_name = f"coarse_tuning_candidate_{i}"
            self.data_access_layer.create_with_figure(message, fig, fig_name)
            plt.close()

        candidates = self.on_analysis_end(candidates_infos)
        return candidates

    def prepare_measurement(self):
        self.station['V_SD'](self.configs["bias_voltage"])
        self.station.awg.stop()

    def perform_measurements(self):
        self.on_measurement_start()
        self.prepare_measurement()
        lower_bound = self.current_candidate.info["lower_bound"]
        upper_bound = self.current_candidate.info["upper_bound"]

        initial_params = {
            "V_L": (lower_bound[0], upper_bound[0]),
            "V_M": (lower_bound[1], upper_bound[1]),
            "V_R": (lower_bound[2], upper_bound[2]),
        }
        path_learner = "models/export.pkl"
        path_checkpoint = "best_model_weights"
        switch_model = FastAISwitchModel(path_learner, path_checkpoint)

        results = run_base_separation_optimisation(
            self.station,
            initial_params=initial_params,
            data_access_layer=self.data_access_layer,
            switch_model= switch_model,
            **self.configs["optim_settings"],
        )

        name_results = timestamp_files() + "_results_coarse_tuning"
        self.data_access_layer.save_data({name_results: results})
        self.current_candidate.data_identifiers["name_results"] = name_results
        self.on_measurement_end()
