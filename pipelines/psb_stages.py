import logging
import os
import pickle
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
from bias_triangle_detection.alt_PSB_detection import PSB_detector_new
from bias_triangle_detection.btriangle_properties import detect_base_alt_slope
from matplotlib import pyplot as plt
from qcodes import Parameter, load_by_guid
from qcodes.dataset import do2d
from skimage.transform import resize

from experiment_control.init_basel import detuning
from helper_functions.data_access_layer import DataAccess
from pipelines.base_stages import BaseStage, Candidate
from pipelines.utils import (
    draw_boxes_and_preds,
    plot_danon_qcodes_with_overview,
    plot_psb_qcodes_with_overview,
    plot_qcodes_data,
    plot_qcodes_with_overview,
    ramp_magnet_before_msmt,
)
from signal_processing import EnsembleClassifier
from signal_processing.danon_gap_detector.danon_gap_detector import DanonGapDetector


from qcodes_addons.AWGhelp import PulseParameter
from qcodes_addons.doNdAWG import do1dAWG, do2dAWG, init_Rabi
from qcodes_addons.Parameterhelp import GateParameter, VirtualGateParameter
from scipy.signal import find_peaks
import scipy.ndimage
from bias_triangle_detection.btriangle_properties import detect_base_alt


DEFAULT_SCORE = 1.0

def score_separation(gray_orig: np.ndarray, masks: np.ndarray, direction: str = 'down'):
    base, corner_pts, _ = detect_base_alt(gray_orig, masks, direction)
    # Gets the short_side of the quadrilateral
    short_side = list(set(corner_pts) - set(tuple(b) for b in base))
    if len(short_side) != 2:
        return 1, None, None, None
    # sort the short_side in such a way that the segments from short_side[i] to base[i] correspond to the two sides of the quadrilateral
    short_side.sort(key = lambda p: np.linalg.norm(p - base[0]))

    # Loops through all the segments from the short_side to the base and gets the average value of the pixels in the segment
    n_steps = 100
    z = []
    for line in np.linspace(np.ravel(short_side), np.ravel(base), n_steps):
        z.append(get_values_from_segment(gray_orig, line).mean())
    z = np.array(z)

    # It should find two peaks and a dip in the middle, corresponding to the separation location
    peaks, _ = find_peaks(z)
    score = DEFAULT_SCORE
    min_loc = None
    if len(peaks) >= 2:
        peaks = peaks[-2:] #take the two rightmost peaks
        min_loc = np.argmin(z[peaks[0]:peaks[1]+1]) + peaks[0]
        # we score by the relative difference wrt the lowest peak
        score = min(z[peaks]/z[min_loc])
    return score, z, peaks, min_loc

def get_values_from_segment(image: np.ndarray, segment: List, n_steps = 100) -> np.ndarray:
    """ Get values from a segment in an image"""
    x0, y0, x1, y1 = segment
    x, y = np.linspace(x0, x1, n_steps), np.linspace(y0, y1, n_steps)
    zi = scipy.ndimage.map_coordinates(image, np.vstack((y,x)))
    return zi


class PSBViaWideScan(BaseStage):
    """
    A class used to represent the wide scan procedure.

    Attributes
    ----------
    station : qcodes.Station
        An object representing the experiment setup.
    experiment_name : str
        Name of the experiment.
    qcodes_parameters : dict
        A dictionary of qcodes parameters used in this stage.
    configs : dict
        A dictionary of configurations for the stage.
    data_access_layer : DataAccess
        An object of DataAccess class for interacting with data.

    Methods
    -------
    prepare_measurement():
        Prepares the instruments for measurement.
    investigate(candidate: Candidate):
        Investigates a given candidate.
    perform_measurements():
        Performs the necessary measurements.
    determine_candidates():
        Determines the candidates for the next stage of experiments.
    """

    def __init__(
        self,
        station: qcodes.Station,
        experiment_name: str,
        qcodes_parameters: Dict[str, Parameter],
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Wdscn"  # needs to be fairly short because of file system limitations
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters
        self.high_magnetic_field = configs["high_magnetic_field"]
        self.low_magnetic_field = configs["low_magnetic_field"]
        self.magnetic_field = float(self.high_magnetic_field)

        self.data_low_magnet_xarray = None
        self.data_high_magnet_xarray = None

        self.bias_direction = None

    def prepare_measurement(self) -> None:
        """
        Prepares the station for measurement by ramping the magnet and stopping the awg.

        Returns
        -------
            None
        """
        ramp_magnet_before_msmt(self.station['IPS'], self.magnetic_field)
        self.station['awg'].stop()

    def investigate(self, candidate: Candidate) -> None:
        """
        Investigates a given candidate for the experiment.

        Parameters
        ----------
            candidate : Candidate
                The candidate to be investigated.

        Returns
        -------
            None
        """
        self.add_candidate(candidate)

        # checking if data had been taken, useful after a crash
        self.load_state()

        self.bias_direction = candidate.info["bias_direction"]

        if not self.current_candidate.data_taken or self.configs["force_measurement"]:
            self.perform_measurements()
        else:
            print(f"Data already taken for {self.name} node")
            print(f"Loading data for {self.name} node")
            msmt_id_high_magnet = self.current_candidate.data_identifiers[
                "high_magnet_measurement"
            ]
            msmt_id_low_magnet = self.current_candidate.data_identifiers[
                "low_magnet_measurement"
            ]
            data_handle = load_by_guid(msmt_id_high_magnet)
            self.data_high_magnet_xarray = data_handle.to_xarray_dataset()
            data_handle_lm = load_by_guid(msmt_id_low_magnet)
            self.data_low_magnet_xarray = data_handle_lm.to_xarray_dataset()

        resulting_candidates = self.current_candidate.resulting_candidates

        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]
        self.bias_direction = self.current_candidate.info["bias_direction"]
        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = resulting_candidates
        print(f"{len(candidates)} candidates found")
        # print(f"candidates found: {candidates}")
        self.state.received_candidates[
            self.state.current_candidate_idx
        ].resulting_candidates = candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def perform_measurements(self) -> None:
        """
        Performs the necessary measurements for the experiment and saves the data.

        Returns
        -------
            None
        """
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()
        print(f"Ramping magnet to {self.high_magnetic_field} T")
        self.magnetic_field = float(self.high_magnetic_field)
        self.prepare_measurement()

        try:
            lp = self.current_candidate.info["left_plunger_voltage"]
            rp = self.current_candidate.info["right_plunger_voltage"]
        except:
            lp = self.configs["left_plunger_voltage"]
            rp = self.configs["right_plunger_voltage"]
        window_rp = self.configs["window_right_plunger"]
        window_lp = self.configs["window_left_plunger"]
        resolution = self.configs["resolution"]

        wait_time_slow_axis = self.configs["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        rp_start = rp - window_rp / 2
        rp_end = rp + window_rp / 2

        lp_start = lp - window_lp / 2
        lp_end = lp + window_lp / 2

        n_px_rp = int(window_rp / resolution)
        n_px_lp = int(window_lp / resolution)

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp}, VLP {lp} "
            f"with windowsize {window_rp, window_lp} "
            f"and number of msmts {n_px_rp, n_px_lp}"
        )

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
            measurement_name='wide_shot_high_magnet'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["high_magnet_measurement"] = msmt_id
        self.data_high_magnet_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_data(self.data_high_magnet_xarray)

        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        print(f"Ramping magnet to {self.low_magnetic_field} T")
        self.magnetic_field = float(self.low_magnetic_field)
        self.prepare_measurement()

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp}, VLP {lp} "
            f"with windowsize {window_rp, window_lp} "
            f"and number of msmts {n_px_rp, n_px_lp}"
        )

        data_handle_lm = do2d(
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
            measurement_name='wide_shot_low_magnet'
        )
        data_handle_lm = data_handle_lm[0]

        msmt_id_low_magnet = data_handle_lm.guid
        self.current_candidate.data_identifiers[
            "low_magnet_measurement"
        ] = msmt_id_low_magnet
        self.data_low_magnet_xarray = data_handle_lm.to_xarray_dataset()

        fig = plot_qcodes_data(data_handle_lm.to_xarray_dataset())

        message = (
            f"2d scan done, low magnetic field, measurement id: {msmt_id_low_magnet}"
        )
        filename = f"stab_diagram_msmst_id_{msmt_id_low_magnet}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.data_access_layer.save_data(
            {
                "wide_shot_scan": {
                    "xarray": self.data_high_magnet_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )
        self.on_measurement_end()
    def cutout(self, img: np.ndarray, blob: np.ndarray, sidelength: int = 50) -> np.ndarray:
        """
        Cuts out a square region from the image centered around the blob.

        Args:
            img (np.ndarray): The input image.
            blob (np.ndarray): The blob around which to cut out.
            sidelength (int, optional): The sidelength of the square to cut out. Defaults to 50.

        Returns:
            np.ndarray: The cut out image.
        """
        x_bottom = (
            int(blob[0] - sidelength / 2) if int(blob[0] - sidelength / 2) >= 0 else 0
        )
        x_top = int(blob[0] + sidelength / 2) if int(blob[0] + sidelength / 2) >= 0 else 0
        y_bottom = (
            int(blob[1] - sidelength / 2) if int(blob[1] - sidelength / 2) >= 0 else 0
        )
        y_top = int(blob[1] + sidelength / 2) if int(blob[1] + sidelength / 2) >= 0 else 0
        img_save = img.copy()
        img = img[x_bottom:x_top, y_bottom:y_top]
        while img.shape[0] != img.shape[1]:
            sidelength -= 1
            img = self.cutout(img_save, blob, sidelength)
        return img
    def score_by_separation(self, data_high_magnet_analysis,
                        locations,
                        configs,
                        sidelength,
                        flip_bias,
                                        ):
        res_h = configs['segmentation_upscaling_res']
        relative_min_area = configs['relative_min_area']
        allow_MET = configs["allow_MET"]
        thr_method = configs["thr_method"]
        denoising = configs["denoising"]
        triangle_direction = "down"
        scores = []

        for location in locations:
            img_leak_cutout = self.cutout(data_high_magnet_analysis, location, sidelength=sidelength)
            if flip_bias:
                img_leak_cutout = img_leak_cutout.T
            if img_leak_cutout.shape[0] == sidelength and img_leak_cutout.shape[1] == sidelength:
                min_area_h = (img_leak_cutout.shape[0] * img_leak_cutout.shape[1] * res_h * res_h) * relative_min_area

                gray_orig, ims, masks = btriangle_detection.triangle_segmentation_alg(img_leak_cutout,
                                                                                      res=res_h,
                                                                                      min_area=min_area_h,
                                                                                      thr_method=thr_method,
                                                                                      denoising=denoising,
                                                                                      allow_MET=allow_MET,
                                                                                      direction=triangle_direction)
                img_leak_cutout = (img_leak_cutout - np.min(img_leak_cutout)) / (np.max(img_leak_cutout) - np.min(img_leak_cutout))
                score, z, peaks, min_loc = score_separation(img_leak_cutout, masks)

                scores.append(score)
            else:
                scores.append(None)


        return scores

    def determine_candidates(self) -> List[Candidate]:
        """
        Determines the candidates for the next stage of experiments based on current data.

        Returns
        -------
            list
                A list of Candidate objects for the next stage of experiments.
        """
        self.on_analysis_start()
        invert_current = self.configs["invert_current"][self.bias_direction]
        if invert_current:
            data_high_magnet_analysis = -self.data_high_magnet_xarray['I_SD'].to_numpy()
            data_low_magnet_analysis = -self.data_low_magnet_xarray['I_SD'].to_numpy()
        else:
            data_high_magnet_analysis = self.data_high_magnet_xarray['I_SD'].to_numpy()
            data_low_magnet_analysis = self.data_low_magnet_xarray['I_SD'].to_numpy()

        plt.imshow(data_high_magnet_analysis)
        plt.title(self.current_candidate.info["bias_direction"])
        plt.colorbar()
        plt.show()

        axes_values = []
        axes_values_names = []
        axes_units = []

        for item, n in dict(self.data_high_magnet_xarray.dims).items():
            axes_values.append(self.data_high_magnet_xarray[item].to_numpy())
            axes_values_names.append(self.data_high_magnet_xarray[item].long_name)
            axes_units.append(self.data_high_magnet_xarray[item].unit)

        offset_px = self.configs["offset_px"]
        (
            anchor,
            peaks_px,
            peaks,
            all_triangles_px,
            all_triangles,
            fig,
        ) = btriangle_location_detection.get_locations(
            data_high_magnet_analysis,
            axes_values[1],
            axes_values[0],
            axes_values_names[1],
            axes_values_names[0],
            return_figure=True,
            plot=True,
            offset_px=offset_px,
        )

        msmt_id = self.current_candidate.data_identifiers["high_magnet_measurement"]

        message = (
            f"bias triangle locations via autocorrelation, original measurement id: {msmt_id}, "
            f"anchor: {anchor}, "
            f"peaks: {peaks}, "
            f"all triangles: {all_triangles}"
        )
        filename = f"_stab_diagram_bias_triangles_detected_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        if self.configs["psb_model"] == "lenet":
            folder_path_to_nn = self.configs['folder_path_to_nn']
            network_names = self.configs['network_names']
            # path_to_nn = "../signal_processing/psb_classifier/saved_networks/lenet_only_sim/"
            psb_classifier = EnsembleClassifier(
                folder_path_to_nn=folder_path_to_nn, network_names=network_names
            )
        else:
            psb_classifier = EnsembleClassifier(
                "../signal_processing/psb_classifier/saved_networks/",
                model_type="resnet18",
            )

        sidelength = np.max(peaks_px)
        if self.current_candidate.info["bias_direction"] == "positive_bias":
            flip_bias = False
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            flip_bias = True
        else:
            raise NotImplementedError

        predictions = psb_classifier.predict_from_large_scan(
            data_low_magnet_analysis,
            data_high_magnet_analysis,
            all_triangles_px,
            sidelength=sidelength,
            flip_bias=flip_bias,
        )
        fig = draw_boxes_and_preds(
            data_high_magnet_analysis,
            data_low_magnet_analysis,
            all_triangles_px,
            predictions,
            sidelength,
        )

        message = f"psb predictions, original measurement id: {msmt_id}, "
        filename = f"_psb_predictions_msmnt_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        pred_cleaned = np.array(predictions)
        pred_cleaned[pred_cleaned == None] = 0

        psb_threshold = self.configs["psb_threshold"]
        print(f'psb_threshold {psb_threshold}')

        separation_configs = self.configs['separation_scoring']
        separation_scores = self.score_by_separation(data_high_magnet_analysis,
                        all_triangles_px,separation_configs,
                        sidelength=sidelength,
                        flip_bias=flip_bias,
                                        )
        fig = draw_boxes_and_preds(
            data_high_magnet_analysis,
            data_low_magnet_analysis,
            all_triangles_px,
            separation_scores,
            sidelength,
            classification_thresh=1.0,
            name='separation scores'
        )

        message = f"score by separation, original measurement id: {msmt_id}, "
        filename = f"_score_separation_msmnt_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        separation_scores_cleaned = np.array(separation_scores)
        separation_scores_cleaned[separation_scores_cleaned == None] = 0
        # There is some dodgy interaction of the tensors of pytorch, numpy, None-types, and lists, this is just a
        # hot fix
        try:
            pred_cleaned = np.array(pred_cleaned)[:,0]
        except IndexError:
            pred_cleaned = np.array(pred_cleaned)
        print(f'separation_scores_cleaned {separation_scores_cleaned}')
        print(f'pred_cleaned {pred_cleaned}')
        candidates_pre_sort = all_triangles[pred_cleaned > psb_threshold]
        print(f'candidates_pre_sort {candidates_pre_sort}')
        candidates_px_pre_sort = all_triangles_px[pred_cleaned > psb_threshold]
        scores_of_candidates = separation_scores_cleaned[pred_cleaned > psb_threshold]
        print(f'scores_of_candidates {scores_of_candidates}')
        idx_sorted = np.argsort(scores_of_candidates)[::-1]
        print(f'idx_sorted {idx_sorted}')
        best_idx = idx_sorted[:self.configs['max_number_candidates']]
        print(f'best_idx {best_idx}')

        candidates = candidates_pre_sort[best_idx]
        print(f'candidates {candidates}')

        candidates_px = candidates_px_pre_sort[best_idx]
        self.data_access_layer.create_or_update_file(
            f"Candidates: {candidates}, candidates_px: {candidates_px}"
        )
        candidates_info = []
        for candidate_voltage, candidate_px in zip(candidates, candidates_px):
            candidate_info = {}
            candidate_info["voltage_location"] = {
                "right_plunger_voltage": candidate_voltage[0],
                "left_plunger_voltage": candidate_voltage[1],
            }
            candidate_info["pixel_location"] = candidate_px
            candidate_info["sidelength"] = sidelength
            candidate_info["bias_direction"] = self.bias_direction
            candidate_info["overview_data_id"] = self.current_candidate.data_identifiers[
                "high_magnet_measurement"
            ]
            candidates_info.append(candidate_info)

        candidates = self.on_analysis_end(candidates_info)
        return candidates

class PSBWideScanPostOptimisation(PSBViaWideScan):
    """
    A class used to represent the wide scan procedure.

    Attributes
    ----------
    station : qcodes.Station
        An object representing the experiment setup.
    experiment_name : str
        Name of the experiment.
    qcodes_parameters : dict
        A dictionary of qcodes parameters used in this stage.
    configs : dict
        A dictionary of configurations for the stage.
    data_access_layer : DataAccess
        An object of DataAccess class for interacting with data.

    Methods
    -------
    prepare_measurement():
        Prepares the instruments for measurement.
    investigate(candidate: Candidate):
        Investigates a given candidate.
    perform_measurements():
        Performs the necessary measurements.
    determine_candidates():
        Determines the candidates for the next stage of experiments.
    """

    def perform_measurements(self) -> None:
        """
        Performs the necessary measurements for the experiment and saves the data.

        Returns
        -------
            None
        """
        self.on_measurement_start()
        # domain = self.current_candidate.info['metadata']['scan_domain']
        lp = self.current_candidate.info["V_LP"]
        rp = self.current_candidate.info["V_RP"]

        print(f"Taking data for {self.name} node")
        self.print_state()
        # if self.current_candidate.info['bias_direction']=='positive_bias':
        #     guid = self.current_candidate.info['metadata']['guid']
        #     self.current_candidate.data_identifiers["high_magnet_measurement"] = guid
        #     data_handle = load_by_guid(guid)
        #     self.data_high_magnet_xarray = data_handle.to_xarray_dataset()
        #
        #     fig = plot_qcodes_data(self.data_high_magnet_xarray)
        #
        #     message = f"2d scan done previously, guid: {guid}"
        #     filename = f"stab_diagram"
        #     self.data_access_layer.create_with_figure(message, fig, filename)
        # else:
        print(f"Ramping magnet to {self.high_magnetic_field} T")
        self.magnetic_field = float(self.high_magnetic_field)
        self.prepare_measurement()

        padding = self.configs['padding']
        rp_start = rp['start']
        rp_stop = rp['stop']
        lp_start = lp['start']
        lp_stop = lp['stop']

        resolution_rp = np.abs(rp_start - rp_stop)/ rp['num_points']
        resolution_lp = np.abs(lp_start - lp_stop)/ lp['num_points']

        resolution = self.configs['resolution']

        rp_start -= padding
        rp_stop += padding
        lp_start -= padding
        lp_stop += padding

        # num_points_rp = int(np.abs(rp_start - rp_stop) / resolution_rp)
        # num_points_lp = int(np.abs(lp_start - lp_stop) / resolution_lp)
        num_points_rp = int(np.abs(rp_start - rp_stop) / resolution)
        num_points_lp = int(np.abs(lp_start - lp_stop) / resolution)
        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp}, VLP {lp}, with padding:"
            f"rp_start {rp_start}, rp_stop {rp_stop}, num_points_rp {num_points_rp}\n"
            f"lp_start {lp_start}, lp_stop {lp_stop}, num_points_lp {num_points_lp}"
        )
        data_handle = do2d(
            self.station['V_RP'],
            rp_start,
            rp_stop,
            num_points_rp,
            rp['delay'],
            self.station['V_LP'],
            lp_start,
            lp_stop,
            num_points_lp,
            lp['delay'],
            self.station['I_SD'],
            show_progress=True,
            measurement_name='wide_shot_high_magnet'
        )
        data_handle = data_handle[0]

        msmt_id_high_magnet = data_handle.guid
        self.current_candidate.data_identifiers[
            "high_magnet_measurement"
        ] = msmt_id_high_magnet
        self.data_high_magnet_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_data(data_handle.to_xarray_dataset())

        message = (
            f"2d scan done, high magnetic field ({self.magnetic_field}T), measurement id: {msmt_id_high_magnet}"
        )
        filename = f"stab_diagram_msmst_id_{msmt_id_high_magnet}"
        self.data_access_layer.create_with_figure(message, fig, filename)


        print(f"Ramping magnet to {self.low_magnetic_field} T")
        self.magnetic_field = float(self.low_magnetic_field)
        self.prepare_measurement()

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp}, VLP {lp} "
        )

        data_handle_lm = do2d(
            self.station['V_RP'],
            rp_start,
            rp_stop,
            num_points_rp,
            rp['delay'],
            self.station['V_LP'],
            lp_start,
            lp_stop,
            num_points_lp,
            lp['delay'],
            self.station['I_SD'],
            show_progress=True,
            measurement_name='wide_shot_low_magnet'
        )
        data_handle_lm = data_handle_lm[0]

        msmt_id_low_magnet = data_handle_lm.guid
        self.current_candidate.data_identifiers[
            "low_magnet_measurement"
        ] = msmt_id_low_magnet
        self.data_low_magnet_xarray = data_handle_lm.to_xarray_dataset()

        fig = plot_qcodes_data(data_handle_lm.to_xarray_dataset())

        message = (
            f"2d scan done, low magnetic field, measurement id: {msmt_id_low_magnet}"
        )
        filename = f"stab_diagram_msmst_id_{msmt_id_low_magnet}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.on_measurement_end()


class HighResPSBClassifier(BaseStage):
    """
    A class to implement a High Resolution Pauli spin blockade Classifier which
    extends the BaseStage functionality.

    Attributes:
    ----------
    station: object
        An object representing the experimental station.
    experiment_name: str
        The name of the experiment.
    qcodes_parameters: Dict
        A dictionary that holds the QCoDeS parameters.
    configs: dict
        A dictionary that holds the configuration parameters.
    data_access_layer: DataAccess
        A DataAccess object providing interface for data I/O operations.
    """

    def __init__(
        self,
        station: qcodes.Station,
        experiment_name: str,
        qcodes_parameters: Dict[str, Parameter],
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "HRPSBClf"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters
        self.high_magnetic_field = configs["high_magnetic_field"]
        self.low_magnetic_field = configs["low_magnetic_field"]
        self.magnetic_field = float(self.high_magnetic_field)

        self.data_low_magnet_xarray = None
        self.data_high_magnet_xarray = None

        self.bias_direction = None

    def prepare_measurement(self) -> None:
        """
        Prepares the measurement by setting up the magnet and stopping any ongoing AWG operation.
        """
        ramp_magnet_before_msmt(self.station['IPS'], self.magnetic_field)
        self.station['awg'].stop()

    def investigate(self, candidate: Candidate) -> None:
        """
        Investigates a candidate. If measurement data is not available or force_measurement
        config is set, it performs measurements. If analysis has not been done or force_analysis
        config is set, it determines new candidates. For each candidate found, it delegates
        the investigation to child nodes.

        Parameters:
        ----------
        candidate: Candidate
            The candidate to investigate.
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
            msmt_id_high_magnet = self.current_candidate.data_identifiers[
                "high_magnet_measurement"
            ]
            msmt_id_low_magnet = self.current_candidate.data_identifiers[
                "low_magnet_measurement"
            ]
            data_handle_hm = load_by_guid(msmt_id_high_magnet)
            self.data_high_magnet_xarray = data_handle_hm.to_xarray_dataset()
            data_handle = load_by_guid(msmt_id_low_magnet)
            self.data_low_magnet_xarray = data_handle.to_xarray_dataset()

        resulting_candidates = self.current_candidate.resulting_candidates
        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = resulting_candidates
        print(f"{len(candidates)} candidates found") #: {candidates}")
        self.current_candidate.resulting_candidates = candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def perform_measurements(self) -> None:
        """
        Performs measurements for a given candidate. It conducts 2D scans at low and high
        magnetic fields and saves the data and the plots. If the measurements are successful,
        it updates the current state's data_taken attribute and saves the state.
        """
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()
        print(f"Ramping magnet to {self.low_magnetic_field} T")
        self.magnetic_field = float(self.low_magnetic_field)
        self.prepare_measurement()

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        # overview_data = self.data_access_layer.load("wide_shot_scan")["xarray"]
        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]

        padding = self.configs['padding']
        resolution = self.configs["resolution"]
        rp_start = self.current_candidate.info['box_scan']['rp_min'] - padding / 2
        rp_end = self.current_candidate.info['box_scan']['rp_max'] + padding / 2

        lp_start = self.current_candidate.info['box_scan']['lp_min'] - padding / 2
        lp_end = self.current_candidate.info['box_scan']['lp_max'] + padding / 2

        n_px_rp = int((rp_end - rp_start) / resolution)
        n_px_lp = int((lp_end - lp_start) / resolution)

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp_start, rp_end}, VLP {lp_start, lp_end} "
            f"and number of msmts {n_px_rp, n_px_lp}"
        )

        # # resolution = self.configs["resolution"]
        #
        wait_time_slow_axis = self.configs["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["wait_time_fast_axis"]


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
            measurement_name='high_res_measurement_low_magnet'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["low_magnet_measurement"] = msmt_id
        self.data_low_magnet_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_with_overview(
            self.data_low_magnet_xarray,
            overview_data,
            location,
            sidelength,
        )

        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        print(f"Ramping magnet to {self.high_magnetic_field} T")
        self.magnetic_field = float(self.high_magnetic_field)
        self.prepare_measurement()

        # self.data_access_layer.create_or_update_file(
        #     f"taking 2d scan around VRP {rp}, VLP {lp} "
        #     f"with windowsize {window_rp, window_lp} "
        #     f"and number of msmts {n_px_rp, n_px_lp}"
        # )
        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around number of msmts {n_px_rp, n_px_lp}"
            f"and resolution {resolution}"
        )

        data_handle_hm = do2d(
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
            measurement_name='high_res_measurement_high_magnet'
        )
        data_handle_hm = data_handle_hm[0]

        msmt_id_high_magnet = data_handle_hm.guid
        self.current_candidate.data_identifiers[
            "high_magnet_measurement"
        ] = msmt_id_high_magnet
        self.data_high_magnet_xarray = data_handle_hm.to_xarray_dataset()

        fig = plot_qcodes_with_overview(
            self.data_high_magnet_xarray,
            overview_data,
            location,
            sidelength,
        )

        message = (
            f"2d scan done, high magnetic field, measurement id: {msmt_id_high_magnet}"
        )
        filename = f"stab_diagram_msmst_id_{msmt_id_high_magnet}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.data_access_layer.save_data(
            {
                "high_res_scan": {
                    "qcodes_arr": data_handle_hm.to_xarray_dataset(),
                    "msmt_id": msmt_id_high_magnet,
                }
            }
        )
        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]
        self.on_measurement_end()

    def check_single_triangle_in_frame(self, data_xarray):
        invert_current = self.configs["invert_current"][self.bias_direction]
        if invert_current:
            data_analysis = -data_xarray["I_SD"].to_numpy()
        else:
            data_analysis = data_xarray["I_SD"].to_numpy()

        res_h = self.configs['segmentation_upscaling_res']
        relative_min_area = self.configs['relative_min_area']
        allow_MET = self.configs["allow_MET"]
        thr_method = self.configs["thr_method"]
        denoising = self.configs["denoising"]
        triangle_direction = "down"

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            pass
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            data_analysis = data_analysis.T
        else:
            raise NotImplementedError

        min_area_h = (data_analysis.shape[0] * data_analysis.shape[1] * res_h * res_h) * relative_min_area

        img, ims, masks = btriangle_detection.triangle_segmentation_alg(data_analysis,
                                                                        res=res_h,
                                                                        min_area=min_area_h,
                                                                        thr_method=thr_method,
                                                                        denoising=denoising,
                                                                        allow_MET=allow_MET,
                                                                        direction=triangle_direction
                                                                        )

        img_new, locations = btriangle_properties.location_by_contour(img, masks)

        multiple_locations_found = False
        if len(locations) > 1:
            multiple_locations_found = True

        return multiple_locations_found

    def determine_candidates(self) -> List[Candidate]:
        """
        Determines the candidates for further investigation based on the configuration and data
        from previous measurements. It updates the current state's analysis_done attribute and
        resulting candidates, then saves the state.

        Returns:
        ----------
        candidates: List[Candidate]
            A list of determined candidates for further investigation.
        """
        self.on_analysis_start()
        invert_current = self.configs["invert_current"][self.bias_direction]
        if invert_current:
            data_high_magnet_analysis = -self.data_high_magnet_xarray["I_SD"].to_numpy()
            data_low_magnet_analysis = -self.data_low_magnet_xarray["I_SD"].to_numpy()
        else:
            data_high_magnet_analysis = self.data_high_magnet_xarray["I_SD"].to_numpy()
            data_low_magnet_analysis = self.data_low_magnet_xarray["I_SD"].to_numpy()

        axes_values = []
        axes_values_names = []
        axes_units = []

        for item, n in dict(self.data_high_magnet_xarray.dims).items():
            axes_values.append(self.data_high_magnet_xarray[item].to_numpy())
            axes_values_names.append(self.data_high_magnet_xarray[item].long_name)
            axes_units.append(self.data_high_magnet_xarray[item].unit)

        high_magnet_multiple_location = self.check_single_triangle_in_frame(self.data_high_magnet_xarray)
        low_magnet_multiple_location = self.check_single_triangle_in_frame(self.data_low_magnet_xarray)
        method = 'segmentation' if 'classification_method' not in self.configs else self.configs['classification_method']
        if method == 'neural_network':
            if self.configs["psb_model"] == "lenet":
                psb_classifier = EnsembleClassifier(
                    "../signal_processing/psb_classifier/saved_networks/lenet_only_sim/"
                )
            else:
                psb_classifier = EnsembleClassifier(
                    "../signal_processing/psb_classifier/saved_networks/",
                    model_type="resnet18",
                )

            el0 = resize(data_low_magnet_analysis, (100, 100))
            el1 = resize(data_high_magnet_analysis, (100, 100))

            if self.current_candidate.info["bias_direction"] == "positive_bias":
                pass
            elif self.current_candidate.info["bias_direction"] == "negative_bias":
                el0 = el0.T
                el1 = el1.T
            else:
                raise NotImplementedError

            prediction = psb_classifier.predict([[el0, el1]])

            prediction = prediction > self.configs["psb_threshold"]
        elif method == 'segmentation':
            prediction = self.psb_via_segmentation(data_low_magnet_analysis, data_high_magnet_analysis)
        else:
            raise NotImplementedError(f'Unknown method {method}')

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]

        fig = plot_psb_qcodes_with_overview(
            self.data_high_magnet_xarray,
            self.data_low_magnet_xarray,
            overview_data,
            location,
            sidelength,
        )
        filename = "psb_classification"
        self.data_access_layer.create_with_figure(
            f"PSB prediction: {prediction}", fig, filename
        )

        candidate_info = self.current_candidate.info
        candidate_info["unpulsed_msmt_id"] = self.current_candidate.data_identifiers[
            "high_magnet_measurement"
        ]
        candidates_info = [candidate_info]

        # candidates = self.build_candidates(candidates_info)
        if high_magnet_multiple_location or low_magnet_multiple_location:
            # candidates = []
            candidates_info = []
            self.data_access_layer.create_or_update_file(f'Discarding candidate because of '
                                                         f'multiple locations detected: '
                                                         f'high magnet: {high_magnet_multiple_location},'
                                                         f'low magnet: {low_magnet_multiple_location}')

        if not prediction:
            # candidates = []
            candidates_info = []
        candidates = self.on_analysis_end(candidates_info)
        return candidates


    def psb_via_segmentation(self, blocked, unblocked):
        res_h = self.configs['segmentation_upscaling_res']
        relative_min_area = self.configs['relative_min_area']
        allow_MET = self.configs["allow_MET"]
        thr_method = self.configs["thr_method"]
        denoising = self.configs["denoising"]
        triangle_direction = "down"

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            pass
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            unblocked = unblocked.T
            blocked = blocked.T
        else:
            raise NotImplementedError

        min_area_h = (unblocked.shape[0] * unblocked.shape[1] * res_h * res_h) * relative_min_area

        unblocked, ims, masks = btriangle_detection.triangle_segmentation_alg(unblocked,
                                                                              res=res_h,
                                                                              min_area=min_area_h,
                                                                              thr_method=thr_method,
                                                                              denoising=denoising,
                                                                              allow_MET=allow_MET,
                                                                              direction=triangle_direction)
        try:
            base, corner_pts, c_im = detect_base_alt_slope(unblocked, masks, 'down')
        except:
            print(f'Error caught, base detection failed')
            return False

        min_area_h = (blocked.shape[0] * blocked.shape[1] * res_h * res_h) * relative_min_area

        blocked, ims2, masks2 = btriangle_detection.triangle_segmentation_alg(blocked,
                                                                              res=res_h,
                                                                              min_area=min_area_h,
                                                                              thr_method=thr_method,
                                                                              denoising=denoising,
                                                                              allow_MET=allow_MET,
                                                                              direction=triangle_direction)

        slope_tol = self.configs["slope_tol"]
        int_tol = self.configs["int_tol"]
        seg_tol = self.configs["seg_tol"]
        median = self.configs["median"]
        pair, PSB = PSB_detector_new(unblocked,
                                     blocked,
                                     base,
                                     masks,
                                     triangle_direction,
                                     slope_tol=slope_tol,
                                     int_tol=int_tol,
                                     median=median,
                                     seg_tol=seg_tol)
        return PSB

def transform_detuning_px_to_volts(axes_points, axes_values, axes_values_names, res):
    first_det_line_px = np.array(axes_points[0]) // res
    second_det_line_px = np.array(axes_points[1]) // res

    volt_locs_1_1 = [
        axes_values[0][first_det_line_px[0][1]],
        axes_values[0][first_det_line_px[1][1]],
    ]
    volt_locs_1_2 = [
        axes_values[1][first_det_line_px[0][0]],
        axes_values[1][first_det_line_px[1][0]],
    ]

    first_det_line = {
        axes_values_names[0]: volt_locs_1_1,
        axes_values_names[1]: volt_locs_1_2,
    }

    volt_locs_2_1 = [
        axes_values[0][np.clip(second_det_line_px[0][1], 0, len(axes_values[0])-1)],
        axes_values[0][np.clip(second_det_line_px[1][1], 0, len(axes_values[0])-1)],
    ]
    volt_locs_2_2 = [
        axes_values[1][np.clip(second_det_line_px[0][0], 0, len(axes_values[1])-1)],
        axes_values[1][np.clip(second_det_line_px[1][0], 0, len(axes_values[1])-1)],
    ]

    second_det_line = {
        axes_values_names[0]: volt_locs_2_1,
        axes_values_names[1]: volt_locs_2_2,
    }
    return first_det_line, second_det_line
class DanonGapCheck(BaseStage):
    """
    This class implements a method for checking for the presence of the Danon Gap.
    """

    def __init__(
        self,
        station,
        experiment_name: str,
        qcodes_parameters: Dict,
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Danon"
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters

        self.magnetic_field_min = float(configs["magnetic_field_min"])

        self.data_xarray = None

        self.bias_direction = None

    def prepare_measurement(self):
        """
        Prepares the experimental station for measurement by ramping the magnet
        and stopping the arbitrary waveform generator.
        """

        ramp_magnet_before_msmt(self.station['IPS'], self.magnetic_field_min)
        self.station['awg'].stop()

    def investigate(self, candidate: Candidate):
        """
        Investigates a candidate by performing measurements if necessary, loading the
        existing state, analyzing the candidate, and determining new candidates.

        Args:
            candidate (Candidate): The candidate to be investigated.
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
            msmt_id = self.current_candidate.data_identifiers["danon_measurement"]
            data_handle = load_by_guid(msmt_id)
            self.data_xarray = data_handle.to_xarray_dataset()

        unpulsed_msmt_id = candidate.info["unpulsed_msmt_id"]
        unpulsed_msmt_xarray = load_by_guid(unpulsed_msmt_id).to_xarray_dataset()

        self.bias_direction = self.current_candidate.info["bias_direction"]
        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = self.current_candidate.resulting_candidates
        print(f"{len(candidates)} candidates")# found: {candidates}")
        self.current_candidate.resulting_candidates = candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def perform_measurements(self):
        """
        Performs measurements by first determining the detuning line, setting up the
        scan ranges, and then running the measurements. The measurement results are saved
        to the data access layer.
        """
        self.on_measurement_start()
        self.print_state()

        ## First determine detuning line:
        unpulsed_msmt_id = self.current_candidate.info["unpulsed_msmt_id"]
        unpulsed_msmt_xarray = load_by_guid(unpulsed_msmt_id).to_xarray_dataset()
        invert_current = self.configs["invert_current"][self.bias_direction]
        if invert_current:
            data_no_pulse_analysis = -unpulsed_msmt_xarray["I_SD"].to_numpy()
        else:
            data_no_pulse_analysis = unpulsed_msmt_xarray["I_SD"].to_numpy()

        data_without_pulsing = data_no_pulse_analysis
        axes_values = []
        axes_values_names = []
        axes_units = []
        for item in unpulsed_msmt_xarray.dims:
            axes_values.append(unpulsed_msmt_xarray[item].to_numpy())
            axes_values_names.append(unpulsed_msmt_xarray[item].long_name)
            axes_units.append(unpulsed_msmt_xarray[item].unit)

        res = self.configs["segmentation_upscaling_res"]

        allow_MET = self.configs["allow_MET"]
        min_area = self.configs["min_area"]
        thr_method = self.configs["thr_method"]

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            triangle_direction = "down"
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            triangle_direction = "up"
        else:
            raise NotImplementedError

        gray_orig, ims, masks = btriangle_detection.triangle_segmentation_alg(
            data_without_pulsing,
            res=res,
            min_area=min_area,
            thr_method=thr_method,
            allow_MET=allow_MET,
            direction=triangle_direction,
        )

        base, corner_pts, c_im = btriangle_properties.detect_base(
            gray_orig, masks, direction=triangle_direction
        )


        # alternative method (for better detuning line cutoff)

        base_alt, corner_pts_alt, c_im_alt = btriangle_properties.detect_base_alt(
            gray_orig, masks, direction=triangle_direction
        )
        padding_factor = self.configs['padding_factor']
        line_img_alt, axes_points_alt = btriangle_properties.detect_refined_detuning_axis(
            gray_orig, [], base_alt, corner_pts_alt, shift = [0,0],
            padding_factor=padding_factor
        )
        print(f'axes_points_alt {axes_points_alt}')



        if len(axes_points_alt) != 0:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=(12, 12), sharex=True, sharey=True
            )

            ax.imshow(line_img_alt)
            ax.axis("off")
            ax.set_title("Original")

            fig.tight_layout()

            message = f"alternative  detuning_line_determination axes_points,: {axes_points_alt}"
            filename = f"alternative_detuning_line_determination"
            self.data_access_layer.create_with_figure(message, fig, filename)
            plt.close()

            first_det_line_alt, second_det_line_alt = transform_detuning_px_to_volts(axes_points_alt, axes_values,
                                                                         axes_values_names, res)
        else:
            # making detuning line a point, forcing the alternative method
            first_det_line_alt={}
            first_det_line_alt['V_RP'] = [0,0]
            first_det_line_alt['V_LP'] = [0,0]
            second_det_line_alt = {}
            second_det_line_alt['V_RP'] = [0, 0]
            second_det_line_alt['V_LP'] = [0, 0]

        axes_points, axes, line_img = btriangle_properties.detect_detuning_axis(
            gray_orig, base, corner_pts
        )

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(12, 12), sharex=True, sharey=True
        )

        ax.imshow(line_img)
        ax.axis("off")
        ax.set_title("Original")

        fig.tight_layout()

        message = f"detuning_line_determination axes_points, axes: {axes_points, axes}"
        filename = f"detuning_line_determination"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        first_det_line, second_det_line = transform_detuning_px_to_volts(axes_points, axes_values,
                                                                         axes_values_names, res)
        message = f'orig method, first_det_line {first_det_line}\n' \
                  f'alt method, first_det_line_alt {first_det_line_alt}'
        self.data_access_layer.create_or_update_file(message)
        minimum_det_line_length_ratio = self.configs['minimum_det_line_length_ratio']

        original_detuning_line_length = np.linalg.norm([np.diff(first_det_line['V_RP']),
                                                        np.diff(first_det_line['V_LP'])])
        alt_detuning_line_length = np.linalg.norm([np.diff(first_det_line_alt['V_RP']),
                                                   np.diff(first_det_line_alt['V_LP'])])
        ratio = alt_detuning_line_length / original_detuning_line_length

        message = f'original_detuning_line_length {original_detuning_line_length},' \
                  f'alt_detuning_line_length {alt_detuning_line_length},' \
                  f'ratio {ratio},' \
                  f'minimum_det_line_length_ratio {minimum_det_line_length_ratio}'
        self.data_access_layer.create_or_update_file(message)
        if 'force_original_line' in self.configs.keys():
            force_original_line = self.configs['force_original_line']
        else:
            force_original_line = False
        if ratio > minimum_det_line_length_ratio and not force_original_line:
            first_det_line_filtered = deepcopy(first_det_line_alt)
            second_det_line_filtered = deepcopy(second_det_line_alt)
            message = f'using alt method, first_det_line_filtered {first_det_line_filtered}'
            self.data_access_layer.create_or_update_file(message)
        else:
            lp = deepcopy(first_det_line['V_LP'])
            rp = deepcopy(first_det_line['V_RP'])
            [slope, intercept] = detuning(lp[1], rp[1], lp[0], rp[0])
            lp[0] = lp[1] - (lp[1] - lp[0]) * minimum_det_line_length_ratio
            rp[0] = lp[0] * slope + intercept
            first_det_line_filtered = {'V_RP': rp, 'V_LP': lp}
            message = f'falling back on original method, first_det_line_filtered {first_det_line_filtered}'
            self.data_access_layer.create_or_update_file(message)

            lp = deepcopy(second_det_line['V_LP'])
            rp = deepcopy(second_det_line['V_RP'])
            [slope, intercept] = detuning(lp[1], rp[1], lp[0], rp[0])
            lp[0] = lp[1]-(lp[1] - lp[0]) * minimum_det_line_length_ratio
            rp[0] = lp[0] * slope + intercept

            second_det_line_filtered = {'V_RP': rp, 'V_LP': lp}
        lp_m = (np.array(second_det_line_filtered['V_LP']) + np.array(first_det_line_filtered['V_LP'])) / 2
        rp_m = (np.array(second_det_line_filtered['V_RP']) + np.array(first_det_line_filtered['V_RP'])) / 2
        middle_det_line_filtered = {'V_RP': rp_m, 'V_LP': lp_m}

        # lp = first_det_line_filtered["V_LP"]
        # rp = first_det_line_filtered["V_RP"]
        lp=lp_m
        rp=rp_m

        params_det = detuning(lp[1], rp[1], lp[0], rp[0])
        [slope, intercept] = params_det
        if self.bias_direction == "positive_bias":
            lp_start = lp[1] + self.configs["detuning_base_offset"]
        else:
            lp_start = lp[1] - self.configs["detuning_base_offset"]
        lp_end = lp[0]


        rp_start = lp_start * slope + intercept
        rp_end = lp_end * slope + intercept
        first_det_line_filtered["V_LP"] = [lp_end, lp_start]
        first_det_line_filtered["V_RP"] = [rp_end, rp_start]
        fig = plt.figure()
        plt.imshow(
            data_without_pulsing,
            origin="lower",
            aspect="auto",
            extent=[
                axes_values[1].min(),
                axes_values[1].max(),
                axes_values[0].min(),
                axes_values[0].max(),
            ],
        )
        plt.ylabel(axes_values_names[0] + " [" + axes_units[0] + "]")
        plt.xlabel(axes_values_names[1] + " [" + axes_units[1] + "]")

        plt.colorbar()
        volt_locs_1_1 = first_det_line_filtered[axes_values_names[0]]
        volt_locs_1_2 = first_det_line_filtered[axes_values_names[1]]
        plt.scatter(volt_locs_1_2, volt_locs_1_1,c= 'white', label="detuning line")
        plt.legend()

        message = (
            f"detuning_line_determination \n"
            f"{first_det_line_filtered}\n"
        )
        filename = f"detuning_lines_in_v_space"
        self.data_access_layer.create_with_figure(message, fig, filename)

        plt.close()

        lp = first_det_line_filtered["V_LP"]
        rp = first_det_line_filtered["V_RP"]
        ########

        print(f"Taking data for {self.name} node")
        print(f"Ramping magnet to {self.magnetic_field_min}")
        self.prepare_measurement()
        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()
        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]

        resolution_magnet = self.configs["resolution_magnet"]
        resolution_detuning = self.configs["resolution_detuning"]

        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        current_candidate_name = self.state.candidate_names[
            self.state.current_candidate_idx
        ]
        params_det = detuning(lp[1], rp[1], lp[0], rp[0])
        detuning_line = VirtualGateParameter(
            name="det_" + current_candidate_name.replace('-', '_'),
            params=(self.station['V_LP'], self.station['V_RP']),
            set_scaling=(1, params_det[0]),
            offsets=(0, params_det[1]),
        )

        window_lp = abs(lp_start - lp_end)

        n_px_lp = int(window_lp / resolution_detuning)

        magnetic_field_min = self.configs["magnetic_field_min"]
        magnetic_field_max = self.configs["magnetic_field_max"]

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
            show_progress=True,
            measurement_name='danon_gap'
        )
        data_handle = data_handle[0]
        self.data_access_layer.create_or_update_file(
            f"measurement taken with run id {data_handle.guid}"
        )

        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["danon_measurement"] = msmt_id
        self.data_xarray = data_handle.to_xarray_dataset()
        fig = plot_danon_qcodes_with_overview(
            self.data_xarray,
            [[rp_end, rp_start], [lp_end, lp_start]],
            unpulsed_msmt_xarray,
            overview_data,
            location,
            sidelength,
        )
        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"danon_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        self.data_access_layer.save_data(
            {
                "danon": {
                    "xarray": self.data_xarray,
                    "msmt_id": msmt_id,
                }
            }
        )
        self.on_measurement_end()

    def determine_candidates(self) -> List[Candidate]:
        """
        Determines further candidates based on the current measurements and returns them.

        Returns:
            List[Candidate]: A list of new candidates based on the current measurements.
        """
        self.on_analysis_start()

        axes_values = []
        axes_values_names = []
        axes_units = []
        for item in self.data_xarray.dims:
            axes_values.append(self.data_xarray[item].to_numpy())
            axes_values_names.append(self.data_xarray[item].long_name)
            axes_units.append(self.data_xarray[item].unit)

        current = self.data_xarray["I_SD"]

        invert_current = self.configs["invert_current"][self.bias_direction]
        # we want to see a peak, so do opposite of what the flag says
        if not invert_current:
            current = -current

        dg_detector = DanonGapDetector(**self.configs["detector"])
        peaks, fig = dg_detector.get_gap_location(current)

        peak_of_magnet = axes_values[0][peaks]
        print(f"peaks found: {peaks, peak_of_magnet}")
        peak_offset_tolerance = self.configs["peak_offset_tolerance"]
        filename = "danon_gap_msmt"
        gap_found = False
        if peaks == None:
            message = "No peaks found. Assume no PSB."
        else:
            if abs(peak_of_magnet) < peak_offset_tolerance:
                message = (
                    f"Peak found"
                    f"peaks: {peak_of_magnet}, peak_offset_tolerance: {peak_offset_tolerance}"
                )
                gap_found = True
            else:
                message = (
                    f"Peaks found in unexpected place. Assume it is noise."
                    f"peaks: {peak_of_magnet}, peak_offset_tolerance: {peak_offset_tolerance}"
                )

        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()
        self.current_candidate = self.state.received_candidates[
            self.state.current_candidate_idx
        ]


        candidate_info = self.current_candidate.info

        candidate_info["unpulsed_msmt_id"] = self.current_candidate.info[
            "unpulsed_msmt_id"
        ]
        # candidates_info.append(candidate_info)
        candidates_info = [candidate_info]
        if not gap_found:
            candidates_info = []
        candidates = self.on_analysis_end(candidates_info)

        return candidates


class ReCenteringStage(BaseStage):
    def __init__(
        self,
        station: qcodes.Station,
        experiment_name: str,
        qcodes_parameters: Dict[str, Parameter],
        configs: dict,
        data_access_layer: DataAccess,
    ):
        name = "Cntr"  # needs to be fairly short because of file system limitations
        super().__init__(experiment_name, configs, name, data_access_layer)
        self.station = station
        self.qcodes_parameters = qcodes_parameters
        self.magnetic_field = float(configs["magnetic_field"])

        self.data_xarray = None
        self.data_xarray_2 = None
        self.bias_direction = None

    def prepare_measurement(self) -> None:
        ramp_magnet_before_msmt(self.station['IPS'], self.magnetic_field)
        self.station['awg'].stop()

    def investigate(self, candidate: Candidate) -> None:
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
            msmt_id = self.current_candidate.data_identifiers["measurement"]
            data_handle = load_by_guid(msmt_id)
            self.data_xarray = data_handle.to_xarray_dataset()

        resulting_candidates = self.current_candidate.resulting_candidates
        if not self.current_candidate.analysis_done or self.configs["force_analysis"]:
            candidates = self.determine_candidates()
        else:
            candidates = resulting_candidates
        print(f"{len(candidates)} candidates found")#: {candidates}")
        self.current_candidate.resulting_candidates = candidates
        for candidate in candidates:
            print(
                f"Investigating candidate {candidate.name} ({candidate.info}), sending to {self.child_stages[0].name}"
            )
            self.child_stages[0].investigate(candidate)

    def perform_measurements(self) -> None:
        """
        Performs measurements for a given candidate. It conducts 2D scans at low and high
        magnetic fields and saves the data and the plots. If the measurements are successful,
        it updates the current state's data_taken attribute and saves the state.
        """
        self.on_measurement_start()
        print(f"Taking data for {self.name} node")
        self.print_state()
        print(f"Ramping magnet to {self.magnetic_field} T")
        self.prepare_measurement()

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        # overview_data = self.data_access_layer.load("wide_shot_scan")["xarray"]
        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        axes_values = []
        for item, n in dict(overview_data.dims).items():
            axes_values.append(overview_data[item].to_numpy())
        resolution_large_scan = (axes_values[0][-1] - axes_values[0][0]) / len(
            axes_values[0]
        )
        print(f"resolution_large_scan {resolution_large_scan}")

        window_rp = resolution_large_scan * sidelength
        window_lp = resolution_large_scan * sidelength
        resolution = self.configs["resolution"]

        wait_time_slow_axis = self.configs["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        lp = self.current_candidate.info["voltage_location"]["left_plunger_voltage"]
        rp = self.current_candidate.info["voltage_location"]["right_plunger_voltage"]

        rp_start = rp - window_rp / 2
        rp_end = rp + window_rp / 2

        lp_start = lp - window_lp / 2
        lp_end = lp + window_lp / 2

        n_px_rp = int(window_rp / resolution)
        n_px_lp = int(window_lp / resolution)

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp}, VLP {lp} "
            f"with windowsize {window_rp, window_lp} "
            f"and number of msmts {n_px_rp, n_px_lp}"
        )

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
            measurement_name='recentering'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["measurement"] = msmt_id
        self.data_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_with_overview(
            self.data_xarray,
            overview_data,
            location,
            sidelength,
        )

        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        self.on_measurement_end()

    def determine_candidates(self) -> List[Candidate]:
        self.on_analysis_start()
        invert_current = self.configs["invert_current"][self.bias_direction]
        if invert_current:
            data_analysis = -self.data_xarray["I_SD"].to_numpy()
        else:
            data_analysis = self.data_xarray["I_SD"].to_numpy()

        axes_values = []
        axes_values_names = []
        axes_units = []

        for item, n in dict(self.data_xarray.dims).items():
            axes_values.append(self.data_xarray[item].to_numpy())
            axes_values_names.append(self.data_xarray[item].long_name)
            axes_units.append(self.data_xarray[item].unit)

        res_h = self.configs['segmentation_upscaling_res']
        relative_min_area = self.configs['relative_min_area']
        allow_MET = self.configs["allow_MET"]
        thr_method = self.configs["thr_method"]
        denoising = self.configs["denoising"]
        triangle_direction = "down"

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            pass
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            data_analysis = data_analysis.T
        else:
            raise NotImplementedError

        min_area_h = (data_analysis.shape[0] * data_analysis.shape[1] * res_h * res_h) * relative_min_area

        img, ims, masks = btriangle_detection.triangle_segmentation_alg(data_analysis,
                                                                        res=res_h,
                                                                        min_area=min_area_h,
                                                                        thr_method=thr_method,
                                                                        denoising=denoising,
                                                                        allow_MET=allow_MET,
                                                                        direction=triangle_direction
                                                                        )

        img_new, locations, bounding_boxes = btriangle_properties.location_and_box_by_contour(img, masks)
        fig = plt.figure()
        plt.imshow(img_new)

        filename = "recentering_scan_raw"
        self.data_access_layer.create_with_figure(
            f"Recentering, locations {locations}", fig, filename
        )
        plt.close()

        if len(locations) > 1:
            center = np.array(img.shape)/2
            distance_to_center = np.linalg.norm(np.array(locations) - center, axis=-1)
            location = locations[np.argmin(distance_to_center)]
            bounding_box = bounding_boxes[np.argmin(distance_to_center)]

            message = f'multiple locations found: {locations}. Using location {location}'
            self.data_access_layer.create_or_update_file(message)
            location = location//res_h
        else:
            location = locations[0]//res_h
            bounding_box = bounding_boxes[0]


        if self.current_candidate.info["bias_direction"] == "positive_bias":
            rp = axes_values[0][location[1]]
            lp = axes_values[1][location[0]]
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            rp = axes_values[0][location[0]]
            lp = axes_values[1][location[1]]
        else:
            raise NotImplementedError
        print(f'rp {rp}, location[1] {location[1]}, axes_values[0] {axes_values[0]}'
              f' lp {lp}, location[0] {location[0]}, axes_values[1] {axes_values[1]} ')
        cmin, rmin, cmax, rmax = bounding_box
        self.data_access_layer.create_or_update_file(f"bounding_box {bounding_box}")
        if rmin == 0 or rmax == masks.shape[0]-1 or cmin == 0 or cmax == masks.shape[1]-1:
            print("box is touching the edge")
            sidelength, lp, rp, bounding_box = self.do_second_centering_scan(lp_voltage=lp, rp_voltage=rp)
            axes_values = []
            axes_values_names = []
            axes_units = []

            for item, n in dict(self.data_xarray_2.dims).items():
                axes_values.append(self.data_xarray_2[item].to_numpy())
                axes_values_names.append(self.data_xarray_2[item].long_name)
                axes_units.append(self.data_xarray_2[item].unit)
        else:
            print("box is not touching the edge")
            sidelength = max(rmax - rmin, cmax - cmin) // res_h
        cmin, rmin, cmax, rmax = bounding_box
        self.data_access_layer.create_or_update_file(f"masks.shape {masks.shape}\n"
                                                     f"img.shape {img.shape}\n"
                                                     f"cmin {cmin}, rmin {rmin}, cmax {cmax}, rmax {rmax}")

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            lp_min = axes_values[1][cmin // res_h]
            lp_max = axes_values[1][cmax // res_h]
            rp_min = axes_values[0][rmin // res_h]
            rp_max = axes_values[0][rmax // res_h]
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            lp_min = axes_values[1][rmin // res_h]
            lp_max = axes_values[1][rmax // res_h]
            rp_min = axes_values[0][cmin // res_h]
            rp_max = axes_values[0][cmax // res_h]
        self.data_access_layer.create_or_update_file(f"lp_min {lp_min}, lp_max {lp_max},\n"
                                                     f"rp_min {rp_min}, rp_max {rp_max}")

        box = np.array(
            [[lp_min, lp_max, lp_max, lp_min, lp_min], [rp_min, rp_min, rp_max, rp_max, rp_min]]).swapaxes(0, 1)

        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        location = self.current_candidate.info["pixel_location"]
        # sidelength = self.current_candidate.info["sidelength"]

        fig = plot_qcodes_with_overview(
            self.data_xarray,
            overview_data,
            location,
            sidelength,
            mark=(lp, rp),
        )
        filename = "recentering_scan"
        self.data_access_layer.create_with_figure(
            f"Recentering, lp {lp}, rp {rp}", fig, filename
        )

        candidate_info = self.current_candidate.info
        candidate_info["box_scan"] = {}
        candidate_info["box_scan"]["rp_min"] = rp_min
        candidate_info["box_scan"]["rp_max"] = rp_max
        candidate_info["box_scan"]["lp_min"] = lp_min
        candidate_info["box_scan"]["lp_max"] = lp_max
        candidate_info["voltage_location"]["left_plunger_voltage"] = lp
        candidate_info["voltage_location"]["right_plunger_voltage"] = rp
        candidate_info['sidelength'] = sidelength
        candidates_info = [candidate_info]
        candidates = self.on_analysis_end(candidates_info)
        # candidates = self.build_candidates(candidates_info)
        #
        # self.current_candidate.resulting_candidates = candidates
        # self.current_candidate.analysis_done = True
        #
        # self.save_state()
        return candidates

    def do_second_centering_scan(self, lp_voltage, rp_voltage):
        msmt_id_overview = self.current_candidate.info["overview_data_id"]
        data_handle = load_by_guid(msmt_id_overview)
        overview_data = data_handle.to_xarray_dataset()

        # overview_data = self.data_access_layer.load("wide_shot_scan")["xarray"]
        location = self.current_candidate.info["pixel_location"]
        sidelength = self.current_candidate.info["sidelength"]
        axes_values = []
        for item, n in dict(overview_data.dims).items():
            axes_values.append(overview_data[item].to_numpy())
        resolution_large_scan = (axes_values[0][-1] - axes_values[0][0]) / len(
            axes_values[0]
        )
        print(f"resolution_large_scan {resolution_large_scan}")

        window_rp = resolution_large_scan * sidelength
        window_lp = resolution_large_scan * sidelength
        resolution = self.configs["resolution"]

        wait_time_slow_axis = self.configs["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["wait_time_fast_axis"]

        lp = lp_voltage
        rp = rp_voltage

        rp_start = rp - window_rp / 2
        rp_end = rp + window_rp / 2

        lp_start = lp - window_lp / 2
        lp_end = lp + window_lp / 2

        n_px_rp = int(window_rp / resolution)
        n_px_lp = int(window_lp / resolution)

        self.data_access_layer.create_or_update_file(
            f"taking 2d scan around VRP {rp}, VLP {lp} "
            f"with windowsize {window_rp, window_lp} "
            f"and number of msmts {n_px_rp, n_px_lp}"
        )

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
            measurement_name='recentering'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.guid
        self.current_candidate.data_identifiers["redo_measurement"] = msmt_id
        self.data_xarray_2 = data_handle.to_xarray_dataset()

        fig = plot_qcodes_with_overview(
            self.data_xarray_2,
            overview_data,
            location,
            sidelength,
        )

        message = f"redo 2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)

        invert_current = self.configs["invert_current"][self.bias_direction]
        if invert_current:
            data_analysis = -self.data_xarray_2["I_SD"].to_numpy()
        else:
            data_analysis = self.data_xarray_2["I_SD"].to_numpy()

        axes_values = []
        axes_values_names = []
        axes_units = []

        for item, n in dict(self.data_xarray_2.dims).items():
            axes_values.append(self.data_xarray_2[item].to_numpy())
            axes_values_names.append(self.data_xarray_2[item].long_name)
            axes_units.append(self.data_xarray_2[item].unit)

        res_h = self.configs['segmentation_upscaling_res']
        relative_min_area = self.configs['relative_min_area']
        allow_MET = self.configs["allow_MET"]
        thr_method = self.configs["thr_method"]
        denoising = self.configs["denoising"]
        triangle_direction = "down"

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            pass
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            data_analysis = data_analysis.T
        else:
            raise NotImplementedError

        min_area_h = (data_analysis.shape[0] * data_analysis.shape[1] * res_h * res_h) * relative_min_area

        img, ims, masks = btriangle_detection.triangle_segmentation_alg(data_analysis,
                                                                        res=res_h,
                                                                        min_area=min_area_h,
                                                                        thr_method=thr_method,
                                                                        denoising=denoising,
                                                                        allow_MET=allow_MET,
                                                                        direction=triangle_direction
                                                                        )

        img_new, locations, bounding_boxes = btriangle_properties.location_and_box_by_contour(img, masks)
        fig = plt.figure()
        plt.imshow(img_new)

        filename = "recentering_scan_raw"
        self.data_access_layer.create_with_figure(
            f"Recentering, locations {locations}", fig, filename
        )
        plt.close()

        if len(locations) > 1:
            center = np.array(img.shape) / 2
            distance_to_center = np.linalg.norm(np.array(locations) - center, axis=-1)
            location = locations[np.argmin(distance_to_center)]
            bounding_box = bounding_boxes[np.argmin(distance_to_center)]
            message = f'multiple locations found: {locations}. Using location {location}'
            self.data_access_layer.create_or_update_file(message)
            location = location // res_h
        else:
            location = locations[0] // res_h
            bounding_box = bounding_boxes[0]

        if self.current_candidate.info["bias_direction"] == "positive_bias":
            rp = axes_values[0][location[1]]
            lp = axes_values[1][location[0]]
        elif self.current_candidate.info["bias_direction"] == "negative_bias":
            rp = axes_values[0][location[0]]
            lp = axes_values[1][location[1]]
        else:
            raise NotImplementedError
        print(f'rp {rp}, location[1] {location[1]}, axes_values[0] {axes_values[0]}'
              f' lp {lp}, location[0] {location[0]}, axes_values[1] {axes_values[1]} ')

        cmin, rmin, cmax, rmax = bounding_box
        sidelength = max(rmax - rmin, cmax - cmin) // res_h

        return sidelength, lp, rp, bounding_box