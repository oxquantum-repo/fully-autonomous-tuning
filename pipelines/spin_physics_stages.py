import logging
import os
import pickle
import time
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import numpy as np
import qcodes
from bias_triangle_detection import (
    btriangle_detection,
    btriangle_location_detection,
    btriangle_properties,
)
from matplotlib import pyplot as plt
from qcodes import Parameter, load_by_run_spec, load_by_guid
from qcodes.dataset import do1d, do2d
from scipy.signal import find_peaks, peak_widths
from skimage.filters import gaussian
from skimage.transform import resize

from experiment_control.init_basel import detuning
from helper_functions.data_access_layer import DataAccess
from helper_functions.pca import PCA
from pipelines.base_stages import BaseStage, Candidate
from pipelines.utils import (
    draw_boxes_and_preds,
    get_current_timestamp,
    plot_b_v_det_qcodes_with_overview,
    plot_danon_qcodes_with_overview,
    plot_lockin_line_qcodes_with_overview,
    plot_lockin_qcodes_with_overview,
    plot_psb_qcodes_with_overview,
    plot_qcodes_data,
    plot_qcodes_with_overview,
    ramp_magnet_before_msmt,
    start_pulsing, plot_line_qcodes_with_overview, plot_rabi_oscillations_with_overview, get_compensated_gates,
)
from signal_processing import EnsembleClassifier
from signal_processing.bias_triangle_processing.bias_triangle_detection.edsr import extract_edsr_spot
from signal_processing.edsr_classifier.readout_point_detector import (
    ReadoutPointDetector,
)


from qcodes_addons.AWGhelp import PulseParameter
from qcodes_addons.doNdAWG import do1dAWG, do2dAWG, init_Rabi
from qcodes_addons.Parameterhelp import GateParameter, VirtualGateParameter


class ReadoutViaBvsDetuning(BaseStage):
    """
    Stage to determine the readout point via a magnetic field vs detuning measurement.
    """

    def __init__(
        self,
        station: qcodes.Station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "BvsDet"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.magnetic_field_min = None  # float(configs["magnetic_field_min"])

        self.data_low_res_xarray = None
        self.data_high_res_xarray = None
        self.pulsed_msmt_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        """
        Prepares the measurement by setting the appropriate parameters and
        initiating certain processes like ramping the magnet and starting pulsing.
        """
        bias_direction = self.current_candidate.info["bias_direction"]
        burst_time = self.current_candidate.info["burst_time"]
        self.station['VS_freq'](self.configs["freq_vs"])
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field_min)
        # start_pulsing(self.station.awg, self.station['V_RP'], bias_direction, burst_time)
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        self.station['LITC'](self.configs["lockin_tc"])
        self.station['mfli'].sigins[0].autorange(1)  # autoranges the lockin input scale
        plt.close()

    def investigate(self, candidate: Candidate):
        """
        Investigates the provided candidate. It adds the candidate, loads the state,
        performs measurements if required, and performs analysis on the data obtained.

        Args:
            candidate (Candidate): Candidate to be investigated.
        """
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")
            msmt_id = (
                self.current_candidate.data_identifiers[
                    "b_v_det_measurement"
                ]
            )
            data_handle = load_by_guid(msmt_id)
            self.data_xarray = data_handle.to_xarray_dataset()

        pulsed_msmt_id = candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates found: {candidates}")
        self.current_candidate.resulting_candidates = candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)



    def perform_measurements(self):
        """
        Perform measurements for the given node. It prepares the measurement, loads the data,
        and updates the data access layer.
        """
        self.on_measurement_end()
        print(f"Taking data for {self.name} node")
        self.print_state()
        self.magnetic_field_min = float(
            self.current_candidate.info["magnetic_field_min"]
        )
        print(f"Ramping magnet to {self.magnetic_field_min} T")
        self.prepare_measurement()

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        pulsed_msmt_id = self.current_candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        resolution_magnet = self.current_candidate.info["resolution_magnet"]
        resolution_detuning = self.configs["resolution_detuning"]

        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        current_candidate_name = self.state.candidate_names[
            self.state.current_candidate_idx
        ]
        lp = self.current_candidate.info["V_LP"]
        rp = self.current_candidate.info["V_RP"]
        params_det = detuning(lp[1], rp[1], lp[0], rp[0])
        detuning_line = VirtualGateParameter(
            name="det_" + current_candidate_name.replace('-', '_'),
            params=(self.station['V_LP'], self.station['V_RP']),
            set_scaling=(1, params_det[0]),
            offsets=(0, params_det[1]),
        )

        lp_start = lp[1]
        lp_end = lp[0]

        window_lp = abs(lp_start - lp_end)

        n_px_lp = int(window_lp / resolution_detuning)

        magnetic_field_min = self.current_candidate.info["magnetic_field_min"]
        magnetic_field_max = self.current_candidate.info["magnetic_field_max"]

        n_px_magnet = int(
            abs(magnetic_field_max - magnetic_field_min) / resolution_magnet
        )

        ramp_rate = self.station.IPS.sweeprate_field.get()  # T/min
        ramp_time = resolution_magnet / ramp_rate * 60  # s

        wait_time_magnet = ramp_time + self.configs["extra_wait_time_slow_axis"]

        self.data_access_layer.create_or_update_file(
            f"taking 1d scan around VRP {rp}, VLP {lp} "
            f"with windowsize {window_lp} "
            f"and number of msmts {n_px_lp}"
            f"from {magnetic_field_min} to {magnetic_field_max} T with resolution"
            f"{resolution_magnet} T"
        )

        data_handle = do2d(
            self.station.IPS.field_setpoint,
            magnetic_field_min,
            magnetic_field_max,
            n_px_magnet,
            wait_time_magnet,
            detuning_line,
            lp_start,
            lp_end,
            n_px_lp,
            wait_time_fast_axis,
            self.station['I_SD'],
            self.station['R'],
            self.station['Phi'],
            self.station['X'],
            self.station['Y'],
            show_progress=True,
            measurement_name='readout_magnet_vs_detuning'
        )
        data_handle = data_handle[0]
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )

        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["b_v_det_measurement"] = msmt_id
        self.data_xarray = data_handle.to_xarray_dataset()
        fig = plot_b_v_det_qcodes_with_overview(
            self.data_xarray,
            [rp, lp],
            self.pulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"b_v_det_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.data_access_layer.save_data(
            {
                "b_v_det": {
                    "qcodes_arr": self.data_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )
        self.on_measurement_end()

    def determine_candidates(self) -> List:
        """
        Determines the candidates based on the current candidate information and the measurements
        taken. Uses a ReadoutPointDetector to find the readout point and constructs new candidates
        based on this information.

        Returns:
            List: List with candidates.
        """
        self.on_analysis_start()
        candidate = self.current_candidate

        lp = candidate.info["V_LP"]
        rp = candidate.info["V_RP"]
        [slope, intercept] = detuning(lp[1], rp[1], lp[0], rp[0])

        readout_detector = ReadoutPointDetector(**self.configs["detector"])

        data_x = self.data_xarray["LIX"]
        data_y = self.data_xarray["LIY"]

        data_squared = data_x**2 + data_y**2
        readout_point, fig = readout_detector.get_readout_point(data_squared.T)
        print(f"readout_point{readout_point}readout_point")
        det_point_px = readout_point[0]
        magnet_px = readout_point[1]

        axes_values = []
        axes_values_names = []
        axes_units = []
        for item in self.data_xarray.dims:
            axes_values.append(self.data_xarray[item].to_numpy())
            axes_values_names.append(self.data_xarray[item].long_name)
            axes_units.append(self.data_xarray[item].unit)

        msmt_id = self.current_candidate.data_identifiers["b_v_det_measurement"]
        filename = f"b_vs_det_msmt_id_{msmt_id}"

        if readout_point[0] is None:
            candidates_info = []
            message = f"No readout spot found."
            self.data_access_layer.create_with_figure(message, fig, filename)
            plt.close()
        else:
            det_point_voltage_lp = axes_values[1][det_point_px]
            magnet_tesla = axes_values[0][magnet_px]

            det_point_voltage_rp = intercept + slope * det_point_voltage_lp

            message = f"det_point_voltage_lp {det_point_voltage_lp}, det_point_voltage_rp {det_point_voltage_rp}, magnet_tesla {magnet_tesla}"
            self.data_access_layer.create_with_figure(message, fig, filename)
            plt.close()

            candidate_info = {
                "magnetic_field": magnet_tesla,
                "left_plunger_voltage": det_point_voltage_lp,
                "right_plunger_voltage": det_point_voltage_rp,
                "bias_direction": candidate.info["bias_direction"],
                "pixel_location": candidate.info["pixel_location"],
                "sidelength": candidate.info["sidelength"],
                "pulsed_msmt_id": candidate.info["pulsed_msmt_id"],
                "b_v_det_measurement_id": candidate.data_identifiers[
                    "b_v_det_measurement"
                ],
                "overview_data_id": candidate.info["overview_data_id"],
                "burst_time": candidate.info["burst_time"]
            }
            candidates_info = [candidate_info]
            # candidates = self.build_candidates(candidates_info)
        candidates = self.on_analysis_end(candidates_info)
        return candidates


class EDSRCheck(BaseStage):
    """
    A class for EDSR checks. It inherits from the BaseStage class.
    It includes functionality to prepare measurements, investigate a candidate, perform measurements, and determine candidates.
    """

    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "EChck"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters
        # 
        self.data_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        bias_direction = self.current_candidate.info["bias_direction"]
        burst_time = self.burst_time
        self.station['VS_freq'](self.freq_vs)
        print(f'VS frequency set to {self.freq_vs}')
        self.station['V_LP'](self.left_plunger_voltage)
        self.station['V_RP'](self.right_plunger_voltage)
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        self.station['LITC'](self.lockin_tc)
        self.station['mfli'].sigins[0].autorange(1)
        plt.close()
        time.sleep(self.lockin_tc * 3)

    def investigate(self, candidate: Candidate):
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")
            msmt_id = self.current_candidate.data_identifiers["edsr_check_measurement"]
            data_handle = load_by_guid(msmt_id)
            self.data_xarray = data_handle.to_xarray_dataset()

        pulsed_msmt_id = candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates found: {candidates}")
        self.current_candidate.resulting_candidates = candidates

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()
        try:
            optim_readout_point = self.current_candidate.info["config"]
            lp = optim_readout_point['V_LP']
            rp = optim_readout_point['V_RP']
            self.freq_vs = optim_readout_point['freq_vs']
            self.burst_time = optim_readout_point['burst_time_ns']*10**(-9)
            self.right_plunger_voltage = rp
            self.left_plunger_voltage = lp
            magnetic_field_center = self.current_candidate.info["magnetic_field"]
            magnetic_field_window = self.configs["magnetic_field_window"]
            magnetic_field_min = magnetic_field_center - magnetic_field_window / 2
            self.sent_by_optimiser = True
        except:
            lp = self.current_candidate.info["left_plunger_voltage"]
            rp = self.current_candidate.info["right_plunger_voltage"]
            self.freq_vs =self.current_candidate.info["freq_vs"]
            self.burst_time =self.current_candidate.info["burst_time"]
            self.right_plunger_voltage = rp
            self.left_plunger_voltage = lp

            magnetic_field_center = self.current_candidate.info["magnetic_field"]
            magnetic_field_window = self.configs["magnetic_field_window"]
            magnetic_field_min = magnetic_field_center - magnetic_field_window / 2
            self.sent_by_optimiser = False

        self.magnetic_field = magnetic_field_min
        self.lockin_tc = self.configs["lockin_tc"]
        print(f"Ramping magnet to {self.magnetic_field} T")
        self.prepare_measurement()
        #
        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        pulsed_msmt_id = self.current_candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        # take another low res pulsed scan to check if bias triangles are still there
        # get ranges from pulsed scan
        axes_values = {}

        for item, n in dict(self.pulsed_msmt_xarray.dims).items():
            axes_values[self.pulsed_msmt_xarray[item].long_name] = (self.pulsed_msmt_xarray[item].to_numpy())

        rp_start = np.min(axes_values['V_RP'])
        rp_end = np.max(axes_values['V_RP'])

        lp_start = np.min(axes_values['V_LP'])
        lp_end = np.max(axes_values['V_LP'])
        resolution = self.configs['switch_check_scan']["resolution"]

        n_px_rp = int((rp_end - rp_start) / resolution)
        n_px_lp = int((lp_end - lp_start) / resolution)

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp_start, rp_end}, VLP {lp_start, lp_end} "
            f"and number of msmts {n_px_rp, n_px_lp}"
            f"and resolution {resolution}"
        )

        wait_time_slow_axis = self.configs['switch_check_scan']["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs['switch_check_scan']["wait_time_fast_axis"]

        data_handle = do2d(
            self.station['V_RP'],
            rp_start,
            rp_end,
            n_px_rp,
            wait_time_slow_axis,
            self.station['V_LP'],
            lp_start,
            lp_end,
            n_px_lp,
            wait_time_fast_axis,
            self.station['I_SD'],
            show_progress=True,
            measurement_name='switch_check'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["switch_check"] = msmt_id

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(data_handle.to_xarray_dataset()['I_SD'], origin='lower')
        axs[0].set_title('switch check')
        axs[1].imshow(self.pulsed_msmt_xarray['I_SD'], origin='lower')
        axs[1].set_title('original')
        message = f"switch check done"
        filename = f"switch_check"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()

        #do2d changed plunger locations, prepare measurement again
        self.prepare_measurement()

        # magnetic_field_min = self.configs['EDSR']['min_magnetic_field']
        magnetic_field_max = magnetic_field_center + magnetic_field_window / 2
        resolution_magnet = self.configs["resolution_magnet"]

        n_px_magnet = int(
            abs(magnetic_field_max - magnetic_field_min) / resolution_magnet
        )

        ramp_rate = self.station.IPS.sweeprate_field.get()  # T/min
        ramp_time = resolution_magnet / ramp_rate * 60  # s

        wait_time_magnet = ramp_time + self.lockin_tc

        self.data_access_layer.create_or_update_file(
            f"taking EDSR check scan at VRP {rp}, VLP {lp} "
            f"from {magnetic_field_min} to {magnetic_field_max} T with resolution"
            f"{resolution_magnet} T"
        )

        data_handle = do1d(
            self.station.IPS.field_setpoint,
            magnetic_field_min,
            magnetic_field_max,
            n_px_magnet,
            wait_time_magnet,
            self.station['I_SD'],
            self.station['R'],
            self.station['Phi'],
            self.station['X'],
            self.station['Y'],
            show_progress=True,
            measurement_name='edsr_check'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.to_xarray_dataset().guid
        self.data_xarray = data_handle.to_xarray_dataset()
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )

        fig = plot_lockin_line_qcodes_with_overview(
            data_handle.to_xarray_dataset(),
            [rp, lp],
            self.pulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"EDSR check 2d scan done, measurement id: {msmt_id}"
        filename = f"edsr_check_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()

        self.current_candidate.data_identifiers["edsr_check_measurement"] = msmt_id
        self.on_measurement_end()

    def determine_candidates(self):
        self.on_analysis_start()
        try:
            optim_readout_point = self.current_candidate.info["config"]
            self.sent_by_optimiser = True
        except:
            self.sent_by_optimiser = False

        axes_values = []
        axes_values_names = []
        axes_units = []
        for item in self.data_xarray.dims:
            axes_values.append(self.data_xarray[item].to_numpy())
            axes_values_names.append(self.data_xarray[item].long_name)
            axes_units.append(self.data_xarray[item].unit)

        data_x = self.data_xarray["LIX"].to_numpy()
        data_y = self.data_xarray["LIY"].to_numpy()
        data_squared = np.sqrt(data_x**2 + data_y**2)
        smoothened_squared = gaussian(data_squared, sigma=self.configs["sigma"])
        plt.plot(smoothened_squared)
        plt.show()
        smoothened_squared = (smoothened_squared - smoothened_squared.min()) / (
            smoothened_squared.max() - smoothened_squared.min()
        )

        peaks, _ = find_peaks(
            smoothened_squared, prominence=self.configs["prominence"]
        )
        if len(peaks) > 0:
            highest_peak_index = peaks[np.argmax(smoothened_squared[peaks])]
            widths, height, left_ips, right_ips = peak_widths(smoothened_squared, [highest_peak_index], rel_height=0.5)
        else:
            pass
        fig = plt.figure()
        plt.plot(axes_values[0], smoothened_squared)
        plt.xlabel(axes_values_names[0])
        plt.ylabel("lockin norm")
        plt.scatter(axes_values[0][peaks], smoothened_squared[peaks], label="peaks")
        # plt.show()
        peak_of_magnet = axes_values[0][peaks]
        print(f"peaks found: {peaks, peak_of_magnet}")
        peak_offset_tolerance = self.configs["peak_offset_tolerance"]
        center_of_scan = (axes_values[0].max() + axes_values[0].min()) / 2
        filename = "edsr_check_msmt"
        peak_found = False
        width_in_T = None
        if len(peaks) > 0:
            plt.hlines(
                y=height,
                xmin=axes_values[0][int(left_ips)],
                xmax=axes_values[0][int(right_ips)],
            )
            plt.legend()
            if any(
                abs(np.array(peak_of_magnet) - center_of_scan) < peak_offset_tolerance
            ) and not self.sent_by_optimiser:
                width = widths[
                    abs(np.array(peak_of_magnet) - center_of_scan)
                    < peak_offset_tolerance
                ][0]
                width_in_T = width * self.configs["resolution_magnet"]
                message = (
                    f"Peak found"
                    f"peaks: {peak_of_magnet}, believed peak: {center_of_scan}, difference: {abs(np.array(peak_of_magnet) - center_of_scan)}"
                    f", peak_offset_tolerance: {peak_offset_tolerance}\n"
                    f"width: {width}, width_in_T: {width_in_T}"
                )
                peak_found = True
            elif self.sent_by_optimiser:
                width = widths[0]
                width_in_T = width * self.configs["resolution_magnet"]
                message = (
                    f"Peak found"
                    f"peaks: {peak_of_magnet}, , believed peak: {center_of_scan}, difference: {abs(np.array(peak_of_magnet) - center_of_scan)}"
                    f", peak_offset_tolerance: {peak_offset_tolerance}\n"
                    f"width: {width}, width_in_T: {width_in_T}"
                )
                peak_inside_tolerance = abs(np.array(peak_of_magnet) - center_of_scan) < peak_offset_tolerance
                print(f'peak_inside_tolerance: {peak_inside_tolerance}')
                if any(peak_inside_tolerance):
                    peak_found = True
                    peak_of_magnet = peak_of_magnet[peak_inside_tolerance][0]
                    print(f'peak_of_magnet: {peak_of_magnet}')
                else:
                    peak_found = False
                    message += '\n Assume noise'
            else:
                message = (
                    f"Peaks found in unexpected place. Assume it is noise."
                    f"peaks: {peak_of_magnet}, peak_offset_tolerance: {peak_offset_tolerance}"
                )
        else:
            message = "No peaks found. Assume no readout."

        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()

        candidate_info = self.current_candidate.info
        candidate_info["pulsed_msmt_id"] = self.current_candidate.info["pulsed_msmt_id"]
        candidate_info["width_in_T"] = width_in_T
        candidate_info["magnetic_field"] = peak_of_magnet
        candidates_info = [candidate_info]

        # candidates = self.build_candidates(candidates_info)
        self.peak_found = peak_found
        if not peak_found:
            candidates_info = []
        candidates = self.on_analysis_end(candidates_info)
        return candidates


class RabiAndEDSR(BaseStage):
    """
    Perform measurements of an EDSR line scan and Rabi chevron scan.
    """

    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "RabiEDSR"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters


        self.data_edsr_xarray = None
        self.data_rabi_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        bias_direction = self.current_candidate.info["bias_direction"]
        burst_time = self.current_candidate.info["burst_time"]
        self.station['VS_freq'](self.configs["freq_vs"])
        self.station['V_LP'](self.left_plunger_voltage)
        self.station['V_RP'](self.right_plunger_voltage)
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        self.station['LITC'](self.lockin_tc)
        self.station['mfli'].sigins[0].autorange(1)
        plt.close()

    def investigate(self, candidate: Candidate):
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()
        else:
            pass

    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()

        lp = self.current_candidate.info["left_plunger_voltage"]
        rp = self.current_candidate.info["right_plunger_voltage"]

        self.right_plunger_voltage = rp
        self.left_plunger_voltage = lp
        self.magnetic_field = float(self.configs["EDSR"]["min_magnetic_field"])
        self.lockin_tc = self.configs["EDSR"]["lockin_tc"]
        print(f"Ramping magnet to {self.magnetic_field} T")
        self.prepare_measurement()

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        pulsed_msmt_id = self.current_candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        magnetic_field_min = self.configs["EDSR"]["min_magnetic_field"]
        magnetic_field_max = self.configs["EDSR"]["max_magnetic_field"]
        resolution_magnet = self.configs["EDSR"]["resolution_magnet"]

        min_freq_vs = self.configs["EDSR"]["min_freq_vs"]
        max_freq_vs = self.configs["EDSR"]["max_freq_vs"]

        resolution_freq = self.configs["EDSR"]["resolution_freq"]

        wait_time_fast_axis = self.configs["EDSR"]["wait_time_fast_axis"]

        self.lockin_tc = self.configs["EDSR"]["lockin_tc"]

        n_px_magnet = int(
            abs(magnetic_field_max - magnetic_field_min) / resolution_magnet
        )
        n_px_freq = int(abs(max_freq_vs - min_freq_vs) / resolution_freq)

        ramp_rate = self.station.IPS.sweeprate_field.get()  # T/min
        ramp_time = resolution_magnet / ramp_rate * 60  # s

        wait_time_magnet = ramp_time + self.configs["EDSR"]["extra_wait_time_slow_axis"]

        self.data_access_layer.create_or_update_file(
            f"taking EDSR scan at VRP {rp}, VLP {lp} "
            f"from {min_freq_vs} to {max_freq_vs} (res: {resolution_freq})"
            f"from {magnetic_field_min} to {magnetic_field_max} T with resolution"
            f"{resolution_magnet} T"
        )

        data_handle = do2d(
            self.station.IPS.field_setpoint,
            magnetic_field_min,
            magnetic_field_max,
            n_px_magnet,
            wait_time_magnet,
            self.station['VS_freq'],
            min_freq_vs,
            max_freq_vs,
            n_px_freq,
            wait_time_fast_axis,
            self.station['I_SD'],
            self.station['R'],
            self.station['Phi'],
            self.station['X'],
            self.station['Y'],
            show_progress=True,
            measurement_name='edsr_line_scan'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.to_xarray_dataset().guid
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )
        self.data_edsr_xarray = data_handle.to_xarray_dataset()

        fig = plot_lockin_qcodes_with_overview(
            self.data_edsr_xarray,
            [rp, lp],
            self.pulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"EDSR line 2d scan done, measurement id: {msmt_id}"
        filename = f"edsr_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        self.data_access_layer.save_data(
            {
                "edsr": {
                    "qcodes_arr": self.data_edsr_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )

        self.current_candidate.data_identifiers["edsr_measurement"] = msmt_id
        print(f"Taking data for {self.name} node")
        self.right_plunger_voltage = rp
        self.left_plunger_voltage = lp
        magnetic_field_window = (
            self.current_candidate.info["width_in_T"]
            * self.configs["Rabi"]["magnetic_field_window_multiplier"]
        )
        self.magnetic_field = (
            self.current_candidate.info["magnetic_field"] - magnetic_field_window / 2
        )

        self.lockin_tc = self.configs["Rabi"]["lockin_tc"]
        self.prepare_measurement()

        resolution_magnet = self.configs["Rabi"]["resolution_magnet"]

        resolution_burst_time = self.configs["Rabi"]["resolution_burst_time"]

        min_burst_time = self.configs["Rabi"]["min_burst_time"]
        max_burst_time = self.configs["Rabi"]["max_burst_time"]
        dead_burst_time = self.configs["Rabi"]["dead_burst_time"]

        extra_wait_time_slow_axis = self.configs["Rabi"]["extra_wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["Rabi"]["wait_time_fast_axis"]

        b_start = (
            self.current_candidate.info["magnetic_field"] - magnetic_field_window / 2
        )
        b_end = (
            self.current_candidate.info["magnetic_field"] + magnetic_field_window / 2
        )
        self.data_access_layer.create_or_update_file(
            f"taking Rabi scan at VRP {rp}, VLP {lp}\n"
            f"with magnetic field {self.magnetic_field} T\n"
            f"from {b_start} to {b_end} T with resolution {resolution_magnet} T\n"
            f"from {min_burst_time} to {max_burst_time} s with resolution"
            f"{resolution_burst_time} s"
        )

        n_px_b = int(abs(b_start - b_end) / resolution_magnet)
        n_px_tburst = int(abs(max_burst_time - min_burst_time) / resolution_burst_time)

        ramp_rate = self.station.IPS.sweeprate_field.get()  # T/min
        ramp_time = resolution_magnet / ramp_rate * 60  # s
        #
        wait_time_magnet = ramp_time + extra_wait_time_slow_axis

        cb_ro_time = dead_burst_time + max_burst_time

        pp = PulseParameter(
            t_RO=cb_ro_time,  # readout part
            t_CB=cb_ro_time,  # coulomb blockade part
            t_ramp=4e-9,
            t_burst=4e-9,
            C_ampl=0,
            I_ampl=0.3,
            # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
            Q_ampl=0.3,
            # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
            IQ_delay=19e-9,  #
            f_SB=0,  # sideband modulation, can get you better signal or enable 2 qubit gates
            f_lockin=87.77,  # avoid 50Hz noise by avoiding multiples of it
            CP_correction_factor=0.848,
        )  # triangle splitting over C_ampl, how much of the pulse arrives at the sample [in mV/mV]
        bias_direction = self.current_candidate.info["bias_direction"]
        cpg_list = get_compensated_gates(self.station)
        cpg_names = [gate.name for gate in cpg_list]
        if 'V_LP' in cpg_names:
            sign_c_ampl_positive_bias = 1
        else:
            sign_c_ampl_positive_bias = -1

        if bias_direction == "positive_bias":
            pp.C_ampl = sign_c_ampl_positive_bias * 0.025
        elif bias_direction == "negative_bias":
            pp.C_ampl = -1 * sign_c_ampl_positive_bias * 0.025
        else:
            raise NotImplementedError
        # pp.C_ampl = -0.025

        data_handle = do2dAWG(
            "Rabi",
            self.station.IPS.field_setpoint_wait,
            b_start,
            b_end,
            n_px_b,
            wait_time_magnet,
            "t_burst",
            min_burst_time,
            max_burst_time,
            n_px_tburst,
            wait_time_fast_axis,
            self.station['I_SD'],
            self.station['X'],
            self.station['Y'],
            self.station['R'],
            self.station['Phi'],
            pp=pp,
            awg=self.station.awg,
            cgp=get_compensated_gates(self.station),
            show_progress=True,
            measurement_name='rabi'
        )
        data_handle = data_handle[0]
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )

        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["rabi_measurement"] = msmt_id
        self.data_rabi_xarray = data_handle.to_xarray_dataset()
        fig = plot_lockin_qcodes_with_overview(
            self.data_rabi_xarray,
            [rp, lp],
            self.pulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"Rabi 2d scan done, measurement id: {msmt_id}"
        filename = f"rabi_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        self.data_access_layer.save_data(
            {
                "rabi": {
                    "qcodes_arr": self.data_rabi_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )
        self.save_gate_space_locations('Rabi and EDSR')

        self.on_measurement_end()

    def save_gate_space_locations(self, message: str = ''):
        """Save the state of the current stage."""
        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        node_name = self.name + "_" + current_name + ".txt"
        file_name = 'gate_voltages.npy'

        gate_voltages = {'VSD': self.station['V_SD'](),
                         'VL': self.station['V_L'](),
                         'VLP': self.station['V_LP'](),
                         'VM': self.station['V_M'](),
                         'VRP': self.station['V_RP'](),
                         'VR': self.station['V_R']()}

        path = os.path.join(
            "..", "data", "experiments", self.experiment_name, "extracted_data", file_name
        )
        try:
            old_locations = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            old_locations = np.array([])
        new_locations = np.array([node_name, gate_voltages, message, get_current_timestamp(), time.time()])

        np.save(path, np.append(old_locations, new_locations))

class EDSRline(BaseStage):
    """
    Perform measurements of an EDSR line scan and Rabi chevron scan.
    """

    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "EDSRline"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.data_edsr_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        bias_direction = self.current_candidate.info["bias_direction"]
        burst_time = self.burst_time
        self.station['V_LP'](self.left_plunger_voltage)
        self.station['V_RP'](self.right_plunger_voltage)
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        self.station['LITC'](self.lockin_tc)
        self.station['mfli'].sigins[0].autorange(1)
        plt.close()

    def investigate(self, candidate: Candidate):
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")
            msmt_id = self.current_candidate.data_identifiers["edsr_measurement"]
            data_handle = load_by_guid(msmt_id)
            self.data_edsr_xarray = data_handle.to_xarray_dataset()

        pulsed_msmt_id = candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates found: {candidates}")
        self.current_candidate.resulting_candidates = candidates

        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)


    def determine_candidates(self):
        self.on_analysis_start()
        candidate = self.current_candidate

        optim_readout_point = candidate.info["config"]
        previous_freq_vs = optim_readout_point['freq_vs']
        previous_magnet = candidate.info['magnetic_field']
        previous_ratio = previous_magnet/previous_freq_vs


        # measure at specific frequency
        vs_freq_desired = 2.79e9
        print(f'previous_ratio: {previous_ratio}')
        print(f'vs_freq_desired: {vs_freq_desired}')
        if isinstance(previous_ratio, list) and len(previous_ratio)>1:
            previous_ratio = previous_ratio[0]
        # magnet_resulting = magnet_tesla/vs_freq * vs_freq_desired
        magnet_resulting = float(previous_ratio * vs_freq_desired)
        # magnet_resulting = int(candidate.info["magnetic_field"])
        message = f'creating candidate to measure at B: {magnet_resulting}'
        self.data_access_layer.create_or_update_file(message)
        try:
            optim_readout_point = self.current_candidate.info["config"]
            lp = optim_readout_point['V_LP']
            rp = optim_readout_point['V_RP']
            burst_time = optim_readout_point['burst_time_ns'] * 10 ** (-9)
        except:
            lp = self.current_candidate.info["left_plunger_voltage"]
            rp = self.current_candidate.info["right_plunger_voltage"]
            burst_time = self.current_candidate.info["burst_time"]

        candidate_info = {
            # "magnetic_field": magnet_tesla,
            # "vs_freq": vs_freq,
            "magnetic_field": magnet_resulting,
            "vs_freq": vs_freq_desired,
            'width_in_T': candidate.info["width_in_T"],
            "left_plunger_voltage": lp,
            "right_plunger_voltage": rp,
            "bias_direction": candidate.info["bias_direction"],
            "pixel_location": candidate.info["pixel_location"],
            "sidelength": candidate.info["sidelength"],
            "pulsed_msmt_id": candidate.info["pulsed_msmt_id"],
            "overview_data_id": candidate.info["overview_data_id"],
            "burst_time": burst_time
        }
        candidates_info = [candidate_info]
        # candidates = self.build_candidates(candidates_info)
        plt.close()
        candidates = self.on_analysis_end(candidates_info)
        return candidates



    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()
        try:
            optim_readout_point = self.current_candidate.info["config"]
            lp = optim_readout_point['V_LP']
            rp = optim_readout_point['V_RP']
            self.freq_vs = optim_readout_point['freq_vs']
            self.burst_time = optim_readout_point['burst_time_ns'] * 10 ** (-9)
            self.right_plunger_voltage = rp
            self.left_plunger_voltage = lp
            magnetic_field_center = 0.25
            magnetic_field_window = 0.5
            magnetic_field_min = magnetic_field_center - magnetic_field_window / 2
            self.sent_by_optimiser = True
        except:
            lp = self.current_candidate.info["left_plunger_voltage"]
            rp = self.current_candidate.info["right_plunger_voltage"]
            self.freq_vs = self.current_candidate.info["freq_vs"]
            self.burst_time = self.current_candidate.info["burst_time"]
            self.right_plunger_voltage = rp
            self.left_plunger_voltage = lp

            magnetic_field_center = self.current_candidate.info["magnetic_field"]
            magnetic_field_window = self.configs["magnetic_field_window"]
            magnetic_field_min = magnetic_field_center - magnetic_field_window / 2
            self.sent_by_optimiser = False
        # lp = self.current_candidate.info["left_plunger_voltage"]
        # rp = self.current_candidate.info["right_plunger_voltage"]
        #
        # self.right_plunger_voltage = rp
        # self.left_plunger_voltage = lp
        self.magnetic_field = float(self.configs["min_magnetic_field"])
        self.lockin_tc = self.configs["lockin_tc"]
        print(f"Ramping magnet to {self.magnetic_field} T")
        self.prepare_measurement()

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        pulsed_msmt_id = self.current_candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        magnetic_field_min = self.configs["min_magnetic_field"]
        magnetic_field_max = self.configs["max_magnetic_field"]
        resolution_magnet = self.configs["resolution_magnet"]

        min_freq_vs = self.configs["min_freq_vs"]
        max_freq_vs = self.configs["max_freq_vs"]

        resolution_freq = self.configs["resolution_freq"]

        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        self.lockin_tc = self.configs["lockin_tc"]

        n_px_magnet = int(
            abs(magnetic_field_max - magnetic_field_min) / resolution_magnet
        )
        n_px_freq = int(abs(max_freq_vs - min_freq_vs) / resolution_freq)

        ramp_rate = self.station.IPS.sweeprate_field.get()  # T/min
        ramp_time = resolution_magnet / ramp_rate * 60  # s

        wait_time_magnet = ramp_time + self.configs["extra_wait_time_slow_axis"]

        self.data_access_layer.create_or_update_file(
            f"taking EDSR scan at VRP {rp}, VLP {lp} "
            f"from {min_freq_vs} to {max_freq_vs} (res: {resolution_freq})"
            f"from {magnetic_field_min} to {magnetic_field_max} T with resolution"
            f"{resolution_magnet} T"
        )

        data_handle = do2d(
            self.station.IPS.field_setpoint,
            magnetic_field_min,
            magnetic_field_max,
            n_px_magnet,
            wait_time_magnet,
            self.station['VS_freq'],
            min_freq_vs,
            max_freq_vs,
            n_px_freq,
            wait_time_fast_axis,
            self.station['I_SD'],
            self.station['R'],
            self.station['Phi'],
            self.station['X'],
            self.station['Y'],
            show_progress=True,
            measurement_name='edsr_line_scan'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.to_xarray_dataset().guid
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )
        self.data_edsr_xarray = data_handle.to_xarray_dataset()

        fig = plot_lockin_qcodes_with_overview(
            self.data_edsr_xarray,
            [rp, lp],
            self.pulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"EDSR line 2d scan done, measurement id: {msmt_id}"
        filename = f"edsr_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        self.data_access_layer.save_data(
            {
                "edsr": {
                    "qcodes_arr": self.data_edsr_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )

        self.current_candidate.data_identifiers["edsr_measurement"] = msmt_id

        self.save_gate_space_locations('EDSR line')

        self.on_measurement_end()

    def save_gate_space_locations(self, message: str = ''):
        """Save the state of the current stage."""
        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        node_name = self.name + "_" + current_name + ".txt"
        file_name = 'gate_voltages.npy'

        gate_voltages = {'VSD': self.station['V_SD'](),
                         'VL': self.station['V_L'](),
                         'VLP': self.station['V_LP'](),
                         'VM': self.station['V_M'](),
                         'VRP': self.station['V_RP'](),
                         'VR': self.station['V_R']()}

        path = os.path.join(
            "..", "data", "experiments", self.experiment_name, "extracted_data", file_name
        )
        try:
            old_locations = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            old_locations = np.array([])
        new_locations = np.array([node_name, gate_voltages, message, get_current_timestamp(), time.time()])

        np.save(path, np.append(old_locations, new_locations))


class Rabi(BaseStage):
    """
    Perform measurements of an EDSR line scan and Rabi chevron scan.
    """

    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Rabi"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters


        self.data_edsr_xarray = None
        self.data_rabi_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        bias_direction = self.current_candidate.info["bias_direction"]
        burst_time = self.current_candidate.info["burst_time"]
        self.station['VS_freq'](self.current_candidate.info["vs_freq"])
        self.station['V_LP'](self.left_plunger_voltage)
        self.station['V_RP'](self.right_plunger_voltage)
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        self.station['LITC'](self.lockin_tc)
        self.station['mfli'].sigins[0].autorange(1)
        plt.close()

    def investigate(self, candidate: Candidate):
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates found: {candidates}")
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)
    def determine_candidates(self):
        self.on_analysis_start()
        msmt_id = self.current_candidate.data_identifiers["rabi_measurement"]
        data = load_by_guid(msmt_id)
        data = data.to_xarray_dataset()

        key_magnet = list(dict(data.dims).keys())[0]
        key_burst_times = list(dict(data.dims).keys())[1]
        burst_times = data[key_burst_times].to_numpy()
        magnetic_fields = data[key_magnet].to_numpy()

        lix = data['LIX']
        liy = data['LIY']

        pca = PCA(lix, liy)
        from scipy.fft import rfft as _rfft
        from scipy.fft import rfftfreq

        number_of_samples = len(burst_times)
        total_time = burst_times[-1]
        sample_rate = number_of_samples / total_time

        out = [rfftfreq(number_of_samples, 1 / sample_rate), _rfft(pca)]
        frequency, amplitudes = out

        real_amplitudes = []
        for amplitude in amplitudes:
            amplitude = amplitude[1:]
            real_amplitudes.append(amplitude * np.conj(amplitude))
            # plt.plot(frequency[1:],amplitude*np.conj(amplitude))
            # max_amplitudes.append(np.max(amplitude))
        fig = plt.figure()
        plt.imshow(np.real(real_amplitudes),
                   extent=[
                       frequency[1],
                       frequency[-1],
                       magnetic_fields[1],
                       magnetic_fields[-1]
                   ],
                   origin='lower',
                   aspect='auto')
        plt.ylabel('magnetic field')
        plt.xlabel('frequency')
        plt.colorbar(label='amplitude of fft')
        magnetic_field_max_idx, rabi_frequ_idx = np.unravel_index(np.argmax(real_amplitudes),
                                                                  np.array(real_amplitudes).shape)
        magnetic_field_max = magnetic_fields[magnetic_field_max_idx]
        rabi_frequ = frequency[rabi_frequ_idx + 1]
        message = f'magnetic_field_max: {magnetic_field_max}, rabi_frequ:{rabi_frequ}'
        fig_name = 'chevron_analysis'
        self.data_access_layer.create_with_figure(message, fig, fig_name)
        plt.close()

        candidate_info = deepcopy(self.current_candidate.info)

        candidate_info["magnetic_field"] = float(magnetic_field_max)

        candidates = self.on_analysis_end([candidate_info])
        return candidates


    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()

        lp = self.current_candidate.info["left_plunger_voltage"]
        rp = self.current_candidate.info["right_plunger_voltage"]

        self.right_plunger_voltage = rp
        self.left_plunger_voltage = lp
        magnetic_field_window = (
            self.current_candidate.info["width_in_T"]
            * self.configs["magnetic_field_window_multiplier"]
        )
        self.magnetic_field = (
            self.current_candidate.info["magnetic_field"] - magnetic_field_window / 2
        )

        self.lockin_tc = self.configs["lockin_tc"]
        self.prepare_measurement()

        resolution_magnet = self.configs["resolution_magnet"]
        n_px_magnet = self.configs['n_px_magnet']
        resolution_burst_time = self.configs["resolution_burst_time"]

        min_burst_time = self.configs["min_burst_time"]
        max_burst_time = self.configs["max_burst_time"]
        dead_burst_time = self.configs["dead_burst_time"]

        extra_wait_time_slow_axis = self.configs["extra_wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        b_start = (
            self.current_candidate.info["magnetic_field"] - magnetic_field_window / 2
        )
        b_end = (
            self.current_candidate.info["magnetic_field"] + magnetic_field_window / 2
        )
        self.data_access_layer.create_or_update_file(
            f"taking Rabi scan at VRP {rp}, VLP {lp}\n"
            f"with magnetic field {self.magnetic_field} T\n"
            f"from {b_start} to {b_end} T with resolution {resolution_magnet} T or n_px_magnet {n_px_magnet}\n"
            f"from {min_burst_time} to {max_burst_time} s with resolution"
            f"{resolution_burst_time} s"
        )

        # n_px_b = int(abs(b_start - b_end) / resolution_magnet)
        # if n_px_b<10:
        #     n_px_b = 10
        n_px_b = n_px_magnet
        n_px_tburst = int(abs(max_burst_time - min_burst_time) / resolution_burst_time)

        ramp_rate = self.station.IPS.sweeprate_field.get()  # T/min
        ramp_time = resolution_magnet / ramp_rate * 60  # s
        #
        wait_time_magnet = ramp_time + extra_wait_time_slow_axis

        cb_ro_time = dead_burst_time + max_burst_time

        pp = PulseParameter(
            t_RO=cb_ro_time,  # readout part
            t_CB=cb_ro_time,  # coulomb blockade part
            t_ramp=4e-9,
            t_burst=4e-9,
            C_ampl=0,
            I_ampl=0.3,
            # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
            Q_ampl=0.3,
            # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
            IQ_delay=19e-9,  #
            f_SB=0,  # sideband modulation, can get you better signal or enable 2 qubit gates
            f_lockin=87.77,  # avoid 50Hz noise by avoiding multiples of it
            CP_correction_factor=0.848,
        )  # triangle splitting over C_ampl, how much of the pulse arrives at the sample [in mV/mV]
        bias_direction = self.current_candidate.info["bias_direction"]
        cpg_list = get_compensated_gates(self.station)
        cpg_names = [gate.name for gate in cpg_list]
        if 'V_LP' in cpg_names:
            sign_c_ampl_positive_bias = 1
        else:
            sign_c_ampl_positive_bias = -1

        if bias_direction == "positive_bias":
            pp.C_ampl = sign_c_ampl_positive_bias * 0.025
        elif bias_direction == "negative_bias":
            pp.C_ampl = -1 * sign_c_ampl_positive_bias * 0.025
        else:
            raise NotImplementedError
        # pp.C_ampl = -0.025

        data_handle = do2dAWG(
            "Rabi",
            self.station.IPS.field_setpoint_wait,
            b_start,
            b_end,
            n_px_b,
            wait_time_magnet,
            "t_burst",
            min_burst_time,
            max_burst_time,
            n_px_tburst,
            wait_time_fast_axis,
            self.station['I_SD'],
            self.station['X'],
            self.station['Y'],
            self.station['R'],
            self.station['Phi'],
            pp=pp,
            awg=self.station.awg,
            cgp=get_compensated_gates(self.station),
            show_progress=True,
            measurement_name='rabi_chevron'
        )
        data_handle = data_handle[0]
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )
        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["rabi_measurement"] = msmt_id
        self.data_rabi_xarray = data_handle.to_xarray_dataset()

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        pulsed_msmt_id = self.current_candidate.info["pulsed_msmt_id"]
        self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

        fig = plot_lockin_qcodes_with_overview(
            self.data_rabi_xarray,
            [rp, lp],
            self.pulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"Rabi 2d scan done, measurement id: {msmt_id}"
        filename = f"rabi_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        self.data_access_layer.save_data(
            {
                "rabi": {
                    "qcodes_arr": self.data_rabi_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )
        self.save_gate_space_locations('Rabi')

        self.on_measurement_end()

    def save_gate_space_locations(self, message: str = ''):
        """Save the state of the current stage."""
        current_name = self.state.candidate_names[self.state.current_candidate_idx]
        node_name = self.name + "_" + current_name + ".txt"
        file_name = 'gate_voltages.npy'

        gate_voltages = {'VSD': self.station['V_SD'](),
                         'VL': self.station['V_L'](),
                         'VLP': self.station['V_LP'](),
                         'VM': self.station['V_M'](),
                         'VRP': self.station['V_RP'](),
                         'VR': self.station['V_R']()}

        path = os.path.join(
            "..", "data", "experiments", self.experiment_name, "extracted_data", file_name
        )
        try:
            old_locations = np.load(path, allow_pickle=True)
        except FileNotFoundError:
            old_locations = np.array([])
        new_locations = np.array([node_name, gate_voltages, message, get_current_timestamp(), time.time()])

        np.save(path, np.append(old_locations, new_locations))


class RabiOscillations(BaseStage):
    """
    Perform measurements of an EDSR line scan and Rabi chevron scan.
    """

    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "RabiO"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.data_rabi_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        bias_direction = self.current_candidate.info["bias_direction"]
        burst_time = self.current_candidate.info["burst_time"]
        self.station['VS_freq'](self.current_candidate.info["vs_freq"])
        self.station['V_LP'](self.left_plunger_voltage)
        self.station['V_RP'](self.right_plunger_voltage)
        ramp_magnet_before_msmt(self.station.IPS, self.magnetic_field)
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        self.station['LITC'](self.lockin_tc)
        self.station['mfli'].sigins[0].autorange(1)
        plt.close()

    def investigate(self, candidate: Candidate):
        print(f"receiving candidate {candidate}")
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()

        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates found: {candidates}")
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)
    def determine_candidates(self):
        self.on_analysis_start()
        rabi_guids = self.current_candidate.data_identifiers["rabi_measurement"]
        all_data = []
        all_pcas = []
        for guid in rabi_guids:
            data = load_by_guid(guid)
            data = data.to_xarray_dataset()
            x = data['LIX']
            y = data['LIY']
            pca = PCA(x,y)
            all_pcas.append(pca)
            all_data.append(data)

        key = list(dict(data.dims).keys())[0]
        burst_times = data[key].to_numpy()

        pca_averaged = np.mean(all_pcas, axis = 0)
        fig = plt.figure()
        plt.plot(burst_times, pca_averaged)
        plt.xlabel('burst times')
        plt.ylabel('PCA of lock in')
        message = 'averaged rabi oscillations'
        fig_name='averaged_rabi_oscs'
        self.data_access_layer.create_with_figure(message, fig, fig_name)
        plt.close()

        candidates = self.on_analysis_end([])
        return candidates


    def perform_measurements(self):
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()

        lp = self.current_candidate.info["left_plunger_voltage"]
        rp = self.current_candidate.info["right_plunger_voltage"]

        self.right_plunger_voltage = rp
        self.left_plunger_voltage = lp

        self.magnetic_field = self.current_candidate.info["magnetic_field"]

        self.lockin_tc = self.configs["lockin_tc"]
        self.prepare_measurement()

        resolution_burst_time = self.configs["resolution_burst_time"]

        min_burst_time = self.configs["min_burst_time"]
        max_burst_time = self.configs["max_burst_time"]
        dead_burst_time = self.configs["dead_burst_time"]

        self.data_access_layer.create_or_update_file(
            f"taking Rabi osc scan at VRP {rp}, VLP {lp}\n"
            f"with magnetic field {self.magnetic_field} T\n"
            f"from {min_burst_time} to {max_burst_time} s with resolution"
            f"{resolution_burst_time} s"
        )

        n_px_tburst = int(abs(max_burst_time - min_burst_time) / resolution_burst_time)
        wait_time_fast_axis = self.lockin_tc + 0.1

        cb_ro_time = dead_burst_time + max_burst_time

        pp = PulseParameter(
            t_RO=cb_ro_time,  # readout part
            t_CB=cb_ro_time,  # coulomb blockade part
            t_ramp=4e-9,
            t_burst=4e-9,
            C_ampl=0,
            I_ampl=0.3,
            # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
            Q_ampl=0.3,
            # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
            IQ_delay=19e-9,  #
            f_SB=0,  # sideband modulation, can get you better signal or enable 2 qubit gates
            f_lockin=87.77,  # avoid 50Hz noise by avoiding multiples of it
            CP_correction_factor=0.848,
        )  # triangle splitting over C_ampl, how much of the pulse arrives at the sample [in mV/mV]
        bias_direction = self.current_candidate.info["bias_direction"]
        cpg_list = get_compensated_gates(self.station)
        cpg_names = [gate.name for gate in cpg_list]
        if 'V_LP' in cpg_names:
            sign_c_ampl_positive_bias = 1
        else:
            sign_c_ampl_positive_bias = -1

        if bias_direction == "positive_bias":
            pp.C_ampl = sign_c_ampl_positive_bias * 0.025
        elif bias_direction == "negative_bias":
            pp.C_ampl = -1 * sign_c_ampl_positive_bias * 0.025
        else:
            raise NotImplementedError
        self.current_candidate.data_identifiers["rabi_measurement"] = []
        for n in range(self.configs['n_repetitions']):
            data_handle = self.do_measurement_rabi_oscillations(0, min_burst_time,
                                                                max_burst_time,
                                                                n_px_tburst,
                                                                wait_time_fast_axis, pp)

            data_handle = data_handle[0]
            self.data_access_layer.create_or_update_file(
                f"measurement taken with run id {data_handle.guid}"
            )
            msmt_id = data_handle.guid
            self.current_candidate.data_identifiers["rabi_measurement"].append(msmt_id)
            self.data_rabi_xarray = data_handle.to_xarray_dataset()

            msmt_id_overview = self.current_candidate.info["overview_data_id"]
            data_handle = load_by_guid(msmt_id_overview)
            overview_data = data_handle.to_xarray_dataset()

            location = self.current_candidate.info["pixel_location"]
            sidelength = self.current_candidate.info["sidelength"]
            pulsed_msmt_id = self.current_candidate.info["pulsed_msmt_id"]
            self.pulsed_msmt_xarray = load_by_guid(pulsed_msmt_id).to_xarray_dataset()

            fig = plot_rabi_oscillations_with_overview(
                self.data_rabi_xarray,
                [rp, lp],
                self.pulsed_msmt_xarray,
                overview_data,
                location,
                sidelength,
            )
            message = f"Rabi oscillation scan done, measurement id: {msmt_id}"
            filename = f"rabi_msmst_id_{msmt_id}"
            self.data_access_layer.create_with_figure(message, fig, filename)
            plt.close()
            # AWG gets upset if we don't do this
            # self.station.awg.stop()
            time.sleep(60)
        self.on_measurement_end()

    def do_measurement_rabi_oscillations(self, idx_of_try, min_burst_time,
                                         max_burst_time,
                                         n_px_tburst,
                                         wait_time_fast_axis, pp, max_idx=5):
        if idx_of_try >= max_idx:
            raise Exception
        # self.station.awg.stop()
        time.sleep(idx_of_try * 120)
        try:
            bias_direction = self.current_candidate.info["bias_direction"]
            burst_time = self.current_candidate.info["burst_time"]
            cpg_list = get_compensated_gates(self.station)
            start_pulsing(self.station.awg, cpg_list, bias_direction, burst_time)
            data_handle = do1dAWG(
                "Rabi",
                "t_burst",
                min_burst_time,
                max_burst_time,
                n_px_tburst,
                wait_time_fast_axis,
                self.station['X'],
                self.station['Y'],
                self.station['R'],
                self.station['Phi'],
                pp=pp,
                awg=self.station.awg,
                cgp=get_compensated_gates(self.station),
                show_progress=True,
                measurement_name='rabi_oscillations'
            )
        except:
            print(f'problem measuring rabi oscillations, idx {idx_of_try} ')
            data_handle = self.do_measurement_rabi_oscillations(idx_of_try + 1, min_burst_time,
                                                                max_burst_time,
                                                                n_px_tburst,
                                                                wait_time_fast_axis, pp)
        return data_handle