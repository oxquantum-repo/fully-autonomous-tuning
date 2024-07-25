from typing import Dict

import numpy as np
import qcodes
from matplotlib import pyplot as plt
from qcodes import Parameter, load_by_id
from qcodes.dataset import do2d

from helper_functions.data_access_layer import DataAccess
from pipelines.base_stages import BaseStage
from pipelines.utils import plot_qcodes_data
from bias_triangle_detection import (
    btriangle_detection,
    btriangle_location_detection,
    btriangle_properties,
)

class NoiseFloorExtraction():
    """This is not intended to be used as a stage but rather as a util for characterisation at the start"""
    def __init__(
        self,
        station: qcodes.Station,
    ):
        self.station = station
        self.current_at_origin = []
        self.current_at_pinchoff = []
        self.barrier_names = ['V_L', 'V_M', 'V_R']


    def prepare_measurement(self) -> None:
        self.station.awg.stop()

    def get_noise_floor(self, n, pinch_off, bias_voltage_low, bias_voltage_high):
        self.prepare_measurement()
        self.station['V_SD'](bias_voltage_low)

        for param_name, value in zip(self.barrier_names, [0,0,0]):
            self.station[param_name](value)

        # determine current at origin
        c = self.station['I_SD']()
        self.current_at_origin.append({'bias_voltage': bias_voltage_low, 'current': c})

        # ramp to pinch off
        for param_name, value in zip(self.barrier_names, pinch_off):
            self.station[param_name](value)


        # measure current in pinch off n times
        currents=[]
        for i in range(n):
            currents.append(self.station['I_SD']())
        self.current_at_pinchoff.append({'bias_voltage': bias_voltage_low, 'current': np.mean(currents),
                                         'raw': currents,
                                         'current_std': np.std(currents)})

        self.station['V_SD'](bias_voltage_high)
        currents = []
        for i in range(n):
            currents.append(self.station['I_SD']())
        self.current_at_pinchoff.append({'bias_voltage': bias_voltage_high, 'current': np.mean(currents),
                                         'raw': currents,
                                         'current_std': np.std(currents)})

        return self.current_at_origin, self.current_at_pinchoff


class VirtualGateExtraction():
    def __init__(
            self,
            station: qcodes.Station,
            configs: dict,
            data_access_layer: DataAccess
    ):
        self.station = station
        self.barrier_names = ['V_L', 'V_M', 'V_R']
        self.plunger_names = ['V_RP', 'V_LP']
        self.configs = configs
        self.data_access_layer = data_access_layer
    def prepare_measurement(self) -> None:
        self.station.awg.stop()

    def get_virtual_gates(self,lp,rp, window_size, bias_direction, delta_in_V=0.01):
        initial_barrier_settings = {}
        for barrier_name in self.barrier_names:
            initial_barrier_settings[barrier_name] = self.station[barrier_name]()
        # initial_plunger_settings={}
        # for plunger_name in self.plunger_names:
        #     initial_plunger_settings[barrier_name] = self.station[plunger_name]()

        # take initial 2d scan
        initial_id = self.take_2d_scan()

        #

        # vary each barrier individually and take scan
        varied_id = {}
        for barrier_name in self.barrier_names:
            self.station[barrier_name](initial_barrier_settings[barrier_name] + delta_in_V)
            varied_id[barrier_name] = self.take_2d_scan(lp,rp, window_size, bias_direction)
            self.station[barrier_name](initial_barrier_settings[barrier_name])

        # extract location from each scan
        location_initial = self.get_locations(initial_id)
        locations_varied = {}
        for barrier_name in self.barrier_names:
            locations_varied[barrier_name] = self.get_locations(varied_id[barrier_name])

        # construct virtual gate matrix
        couplings = self.get_couplings(location_initial, locations_varied, delta_in_V)

        return couplings

    def get_locations(self, msmt_id, bias_direction):
        data_xarray = load_by_id(msmt_id).to_xarray_dataset()
        invert_current = self.configs["invert_current"][bias_direction]
        if invert_current:
            data_analysis = -data_xarray["I_SD"].to_numpy()
        else:
            data_analysis = data_xarray["I_SD"].to_numpy()

        axes_values = []
        axes_values_names = []
        axes_units = []

        for item, n in dict(data_xarray.dims).items():
            axes_values.append(data_xarray[item].to_numpy())
            axes_values_names.append(data_xarray[item].long_name)
            axes_units.append(data_xarray[item].unit)

        res_h = self.configs['segmentation_upscaling_res']
        relative_min_area = self.configs['relative_min_area']
        allow_MET = self.configs["allow_MET"]
        thr_method = self.configs["thr_method"]
        denoising = self.configs["denoising"]
        triangle_direction = "down"

        if bias_direction == "positive_bias":
            pass
        elif bias_direction == "negative_bias":
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
        fig = plt.figure()
        plt.imshow(img_new)

        filename = "recentering_scan_raw"
        self.data_access_layer.create_with_figure(
            f"Recentering, locations {locations}", fig, filename
        )
        plt.close()

        location = locations[0] // res_h

        if bias_direction == "positive_bias":
            rp = axes_values[0][location[1]]
            lp = axes_values[1][location[0]]
        elif bias_direction == "negative_bias":
            rp = axes_values[0][location[0]]
            lp = axes_values[1][location[1]]
        else:
            raise NotImplementedError
        print(f'rp {rp}, location[1] {location[1]}, axes_values[0] {axes_values[0]}'
              f' lp {lp}, location[0] {location[0]}, axes_values[1] {axes_values[1]} ')


        fig = plot_qcodes_data(
            data_xarray,
            mark=(lp, rp),
        )
        filename = "recentering_scan"
        self.data_access_layer.create_with_figure(
            f"Recentering, lp {lp}, rp {rp}", fig, filename
        )

        return {'V_LP': lp, 'V_RP': rp}

    def get_couplings(self, initial_loc, varied_locs, delta_V):
        coupling_strengths = {}
        for barrier_name, location in varied_locs.items():
            coupling_strengths[barrier_name] = (location - initial_loc[barrier_name])/delta_V
        print(f'coupling strengths: {coupling_strengths}')
        return coupling_strengths
    def take_2d_scan(self,lp,rp, window_size):
        window_rp = window_size
        window_lp = window_size
        resolution = self.configs["resolution"]

        wait_time_slow_axis = self.configs["wait_time_slow_axis"]
        wait_time_fast_axis = self.configs["wait_time_fast_axis"]


        rp_start = rp - window_rp / 2
        rp_end = rp + window_rp / 2

        lp_start = lp - window_lp / 2
        lp_end = lp + window_lp / 2

        n_px_rp = int(window_rp / resolution)
        n_px_lp = int(window_lp / resolution)

        print(
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
            self.station['VLP'],
            lp_start,
            lp_end,
            n_px_lp,
            wait_time_fast_axis,
            self.station['I_SD'],
            show_progress=True,
            measurement_name='virtual_gate_extraction'
        )
        data_handle = data_handle[0]
        msmt_id = data_handle.run_id
        data_xarray = data_handle.to_xarray_dataset()

        fig = plot_qcodes_data(data_xarray)

        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plt.close()

        return msmt_id
