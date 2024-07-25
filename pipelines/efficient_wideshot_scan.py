from typing import Any, Dict, List, Tuple

import qcodes
import toml
from bias_triangle_detection.wideshot_detection import WideshotSampler
from bias_triangle_detection.wideshot_detection.viz import plot_measurement_mask, plot_measurement_histogram, plot_detected_contor_locations
from matplotlib import pyplot as plt

from qcodes import Parameter
from qcodes.dataset import do2d

from experiment_control.init_basel import initialise_experiment
from experiment_control.data_taking_utils import WideShotScanConfig
from experiment_control.dummy_init import initialise_dummy_experiment
from helper_functions.data_access_layer import DataAccess
from helper_functions.file_system_tools import create_folder_structure
from pipelines.base_stages import RootStage
from pipelines.utils import plot_qcodes_data

import xarray as xr
from pipelines.psb_stages import PSBViaWideScan
import os 

class PSBViaEfficientWideScan(PSBViaWideScan):
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
        super().__init__(station, experiment_name, qcodes_parameters, configs, data_access_layer)
        self.measurement_idx = 0

    def perform_measurements(self) -> None:
        """
        Performs the necessary measurements for the experiment and saves the data.

        Returns
        -------
            None
        """
        print(f"Taking data for {self.name} node")
        self.print_state()
        print(f"Ramping magnet to {self.high_magnetic_field} T")
        self.magnetic_field = float(self.high_magnetic_field)
        self.prepare_measurement()

        msmt_id, wideshot_sampler = self._perform_scan()
        self._save_scan_data(wideshot_sampler, msmt_id, "high_magnet")
        self.data_high_magnet_xarray = self._save_scan_data(wideshot_sampler, msmt_id, "high_magnet")

        print(f"Ramping magnet to {self.low_magnetic_field} T")
        self.magnetic_field = float(self.low_magnetic_field)
        self.prepare_measurement()

        msmt_id_low_magnet, wideshot_sampler = self._perform_scan()
        self.data_low_magnet_xarray = self._save_scan_data(wideshot_sampler, msmt_id_low_magnet, "low_magnet")
        self.current_candidate.data_taken = True
        self.save_state()

    def _perform_scan(self) -> Tuple[str, Any, List[Tuple[int,int]]]:
        scan_config = self._configure_scan_range()

        self.data_access_layer.create_or_update_file(
            f"Performing contour scan around VRP {scan_config.rp}, VLP {scan_config.lp} "
            f"for window sized {scan_config.window_rp, scan_config.window_lp} "
        )

        wideshot_sampler = WideshotSampler(
            (scan_config.n_px_lp, scan_config.n_px_rp),
            perform_measurement=lambda point: self._measure_current_at_point(point, scan_config)
        )
        wideshot_sampler.sample(on_contour_measure=lambda idx: None)
        self.measurement_idx += 1
        return f"wideshot_msmt_{self.measurement_idx}", wideshot_sampler

    def _measure_current_at_point(self, point: Tuple[int, int], scan_config) -> float:
        rp_start, lp_start = scan_config.pixel_idx_to_plunger_voltages(point)
        self.VRP(rp_start)
        self.VLP(lp_start)
        current = -self.ISD()
        return current

    def _configure_scan_range(self) -> "WideShotScanConfig":
        return WideShotScanConfig(
            lp=self.current_candidate.info["left_plunger_voltage"],
            rp=self.current_candidate.info["right_plunger_voltage"],
            window_rp = self.configs["window_right_plunger"],
            window_lp = self.configs["window_left_plunger"],
            resolution = self.configs["resolution"]
        )
    
    def _save_scan_data(self, wideshot_sampler: WideshotSampler, msmt_id: str, prefix: str) -> None:
        self.current_candidate.data_qcodes_ids[f"{prefix}_measurement"] = msmt_id
        scan_config = self._configure_scan_range()
        I_SD = xr.DataArray(
            -1.0*wideshot_sampler.data,
            dims=["VLP", "VRP"],
            coords={
                "VLP": scan_config.lp_range,
                "VRP": scan_config.rp_range,
            },
        )
        I_SD['VLP'].attrs['unit'] = "mV"
        I_SD['VRP'].attrs['unit'] = "mV"
        I_SD.__setitem__("long_name", "Measured data by wideshot sampler")
        I_SD.__setitem__("unit", "A")

        data_xarray = xr.Dataset(
            data_vars=dict(
              I_SD=I_SD
            )
        )
        self.data_access_layer.save_data({prefix: wideshot_sampler.data})
        fig = plot_qcodes_data(data_xarray)
        message = f"2d scan done, measurement id: {msmt_id}"
        filename = f"stab_diagram_msmst_id_{msmt_id}"
        self.data_access_layer.create_with_figure(message, fig, filename)
        plot_measurement_mask(wideshot_sampler.binary_data, f"{os.getcwd()}/{prefix}_binary_data.png")
        plot_measurement_mask(wideshot_sampler.measurement_mask, f"{os.getcwd()}/{prefix}_measurement_locations.png")
        plot_measurement_mask(wideshot_sampler.data, f"{os.getcwd()}/{prefix}_data.png")
        return data_xarray
        
if __name__ == "__main__":
    test_mode: bool = True
    if test_mode:
        import qcodes as qc
        from qcodes import (
            ManualParameter,
            initialise_or_create_database_at,
            load_or_create_experiment,
        )
        from qcodes.instrument.parameter import ManualParameter, ScaledParameter
        from qcodes.tests.instrument_mocks import (
            DummyInstrument,
            DummyInstrumentWithMeasurement,
        )
        from qcodes_addons.Parameterhelp import GateParameter

        sample_name = "Butch"  # Sample name
        exp_name = "Qubit_Search"  # Experiment name
        load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

        station = qc.Station()
        dac = DummyInstrument(
            name="LNHR_dac", gates=["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch8"]
        )
        station.add_component(dac)
        DAQ = DummyInstrumentWithMeasurement(name="daq", setter_instr=dac)
        station.add_component(DAQ)
        gain_SD = ManualParameter("gain_SD", initial_value=1 * 1e9, unit="V/A")
        ISD = ScaledParameter(DAQ.v1, name="I_SD", division=gain_SD, unit="A")
        LIX = ScaledParameter(DAQ.v1, name="X", division=gain_SD, unit="A")
        LIY = ScaledParameter(DAQ.v2, name="Y", division=gain_SD, unit="A")
        LIR = ScaledParameter(DAQ.v1, name="R", division=gain_SD, unit="A")
        LIPhi = ScaledParameter(DAQ.v2, name="Phi", division=gain_SD, unit="A")
        value_range = (-3, 3)
        VLP = GateParameter(dac.ch1, name="V_LP", unit="V", value_range=value_range)
        VRP = GateParameter(dac.ch2, name="V_RP", unit="V", value_range=value_range)
        VL = GateParameter(dac.ch3, name="V_L", unit="V", value_range=value_range)
        VM = GateParameter(dac.ch4, name="V_M", unit="V", value_range=value_range)
        VR = GateParameter(dac.ch5, name="V_R", unit="V", value_range=value_range)
        VSD = GateParameter(dac.ch8,
                        name = "V_SD",
                        unit = "V",
                        value_range = (-6, 6),
                        scaling = 103.8,
                        offset = 0)

        field_setpoint = GateParameter(
            dac.ch6, name="field_setpoint", unit="T", value_range=value_range
        )
        components = [ISD, VLP, VRP, VL, VM, VR, LIX, LIY, field_setpoint, LIR, LIPhi]
        for component in components:
            station.add_component(component)
        
        ips = DummyInstrument(name = "IPS")
        class foo:
            def get(self):
                return 1.0 

        ips.sweeprate_field  = foo()  
        ips._get_field_setpoint = lambda: 0.1
        ips.run_to_field = lambda x: 0.1
        station.add_component(ips)
        
        awg = DummyInstrument(name='awg') #, 'TCPIP0::192.168.10.2::INSTR') #driver adapted from 5208 version, setting configs in AWG70000A the same as for 5204 as for 5208
        awg.stop = lambda: None 
        station.add_component(awg)

        VS = DummyInstrument("VS") #, "USB0::0x0AAD::0x0088::110184::INSTR")
        station.add_component(VS)

        qcodes_parameters = {
                "VSD": VSD,
                "VRP": VRP,
                "VLP": VLP,
                "ISD": ISD,
                "LIX": LIX,
                "LIY": LIY,
                "LIR": LIR,
                "LIPhi": LIPhi,
                "VM": VM,
                "VL": VL,
                "VR": VR,
        }

    else:
        (
            station,
            ips,
            awg,
            pp,
            DAQ,
            mfli,
            VS,
            VS_freq,
            VS_phase,
            VS_pwr,
            VS_pulse_on,
            VS_pulse_off,
            VS_IQ_on,
            VS_IQ_off,
            VS_status,
            LIXY,
            LIXYRPhi,
            LIX,
            LIY,
            LIR,
            LIPhi,
            LIPhaseAdjust,
            LIfreq,
            LITC,
            ISD,
            DAQ,
            VS,
            VM,
            VL,
            VLP,
            VR,
            VRP,
            VSD,
        ) = initialise_experiment()
        qcodes_parameters = {
            "VSD": VSD,
            "VRP": VRP,
            "VLP": VLP,
            "ISD": ISD,
            "LIX": LIX,
            "LIY": LIY,
            "LIR": LIR,
            "LIPhi": LIPhi,
            "LITC": LITC,
            "mfli": mfli,
            "VS_freq": VS_freq,
            "VM": VM,
            "VL": VL,
            "VR": VR,
        }

    config_path = "../data/config_files/v3.toml"

    configs = toml.load(config_path)

    experiment_name = "20230707_testing_efficient_wide_scan"
    config_path = "../data/config_files/v3.toml"

    create_folder_structure(experiment_name)
    with open(
            "../data/experiments/" + experiment_name + "/documentation/configs.toml", "w"
    ) as f:
        toml.dump(configs, f)
    data_access_layer = DataAccess(experiment_name)

    scan_stage = PSBViaEfficientWideScan(
        station=station,
        experiment_name=experiment_name,
        qcodes_parameters=qcodes_parameters,
        configs=configs['PSBViaWideScan'],
        data_access_layer=data_access_layer
    )

    root_stage = RootStage(
        experiment_name, qcodes_parameters, configs["RootStage"], data_access_layer
    )
    scan_stage.parent_stages = [root_stage]

    root_stage.child_stages = [scan_stage]
    root_stage.kick_off()
    # scan_stage.perform_measurements()
