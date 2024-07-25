import abc
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union
import pickle

import numpy as np
import qcodes as qc
from bias_triangle_detection.qcodes_db_utils import query_datasets
from bias_triangle_detection.wideshot_detection import WideshotSampler
from bias_triangle_detection.wideshot_detection.viz import (
    create_debug_plots_for_wideshot_sampler,
)
from loguru import logger
from qcodes.dataset import AbstractSweep, DataSetProtocol, LinSweep, Measurement, dond
from qcodes.dataset.data_set_protocol import DataSetProtocol
from qcodes.parameters import ParameterBase
from qcodes.utils.validators import Numbers
from tenacity import retry, stop_after_attempt

from pipelines.utils import ramp_magnet_before_msmt, start_pulsing, get_compensated_gates
from scipy.stats import mode 

EPSILON = 1e-6


@dataclass
class LinSweepConfig:
    start: float
    stop: float
    num_points: int
    delay: float = 0


@dataclass
class WideShotScanConfig:
    rp: float
    lp: float
    window_rp: float
    window_lp: float
    resolution: float

    @property
    def rp_start(self) -> int:
        return self.rp - self.window_rp / 2

    @property
    def rp_end(self) -> int:
        return self.rp + self.window_rp / 2

    @property
    def lp_start(self) -> int:
        return self.lp - self.window_lp / 2

    @property
    def lp_end(self) -> int:
        return self.lp + self.window_lp / 2

    @property
    def n_px_rp(self) -> int:
        return int(self.window_rp / self.resolution)

    @property
    def n_px_lp(self) -> int:
        return int(self.window_lp / self.resolution)

    @property
    def lp_range(self) -> Tuple[float, ...]:
        return np.linspace(self.lp_start, self.lp_end, self.n_px_lp)

    @property
    def rp_range(self) -> Tuple[float, ...]:
        return np.linspace(self.rp_start, self.rp_end, self.n_px_rp)

    def pixel_idx_to_plunger_voltages(
        self, pixel_idx: Tuple[int, int]
    ) -> Tuple[float, float]:
        rp = self.rp_start + pixel_idx[0] * self.resolution
        lp = self.lp_start + pixel_idx[1] * self.resolution
        return rp, lp


@dataclass
class WideShotConfig:
    v_rp_config: Dict[str, Any]
    v_lp_config: Dict[str, Any]

    @property
    def rp(self) -> float:
        return self.v_rp_config["start"] + self.window_rp / 2

    @property
    def lp(self) -> float:
        return self.v_lp_config["start"] + self.window_lp / 2

    @property
    def window_rp(self) -> float:
        return self.v_rp_config["stop"] - self.v_rp_config["start"]

    @property
    def window_lp(self) -> float:
        return self.v_lp_config["stop"] - self.v_lp_config["start"]

    @property
    def resolution_rp(self) -> float:
        return self.window_rp / (self.v_rp_config["num_points"] - 1)

    @property
    def resolution_lp(self) -> float:
        return self.window_lp / (self.v_lp_config["num_points"] - 1)

    @property
    def n_px_lp(self) -> int:
        return self.v_lp_config["num_points"] + 1

    @property
    def n_px_rp(self) -> int:
        return self.v_rp_config["num_points"] + 1

    @property
    def rp_start(self) -> int:
        return self.rp - self.window_rp / 2

    @property
    def rp_end(self) -> int:
        return self.rp + self.window_rp / 2

    @property
    def lp_start(self) -> int:
        return self.lp - self.window_lp / 2

    @property
    def lp_end(self) -> int:
        return self.lp + self.window_lp / 2

    @property
    def n_px_rp(self) -> int:
        return int(self.window_rp / self.resolution_rp) + 1

    @property
    def n_px_lp(self) -> int:
        return int(self.window_lp / self.resolution_lp) + 1

    @property
    def lp_range(self) -> Tuple[float, ...]:
        return np.linspace(self.lp_start, self.lp_end, self.n_px_lp)

    @property
    def rp_range(self) -> Tuple[float, ...]:
        return np.linspace(self.rp_start, self.rp_end, self.n_px_rp)

    def pixel_idx_to_plunger_voltages(
        self, pixel_idx: Tuple[int, int]
    ) -> Tuple[float, float]:
        rp = self.rp_start + pixel_idx[0] * self.resolution_rp
        lp = self.lp_start + pixel_idx[1] * self.resolution_lp
        return rp, lp

    def plunger_voltages_to_pixel_idx(self, rp: float, lp: float) -> Tuple[int, int]:
        rp_idx = int((rp - self.rp_start) / self.resolution_rp)
        lp_idx = int((lp - self.lp_start) / self.resolution_lp)
        return rp_idx, lp_idx


ScanConfig = Dict[
    str, Union[LinSweepConfig, Dict[str, Any]]
]  # param name to scan config plus scan kwargs


def _scan_nd_with_snake(
    station: qc.Station,
    scan_config: ScanConfig,
    prepared_measurement: bool = True,
    debug: bool = False,
    fill_with_threshold: bool = True,
    data_save_path: str = '../data/'
) -> DataSetProtocol:
    assert prepared_measurement, "Please prepare measurement before scanning"
    meas = Measurement(
        station=station,
        name=scan_config.get("scan_kwargs", {}).get("measurement_name", ""),
    )
    setpoints = []
    for param_name in scan_config.get("domain", {}):
        param = station[param_name]
        setpoints.append(param)
        meas.register_parameter(param)
    values_to_measure = []
    for meas_name in scan_config.get("measure", []):
        measure_param = station[meas_name]
        values_to_measure.append(measure_param)
        meas.register_parameter(measure_param, setpoints=tuple(setpoints))
    assert len(values_to_measure) == 1, "Only one measure param is supported"

    if debug:
        task_id_high_res = "cddeb475-85bd-4b68-8359-732e580522c5"
        path = "data/GeSiNW_Qubit_VTI01_Jonas_2.db"
        dataset = query_datasets(path, lambda ds: ds.name.endswith(task_id_high_res))[0]
        dsarr = dataset.to_xarray_dataset()
        rc, lc = list(dsarr.dims)
        name_dic = {"V_RP": rc, "V_LP": lc}
        scan_config["domain"] = {
            "V_LP": {
                "start": dsarr.coords[lc].min().item(),
                "stop": dsarr.coords[lc].max().item(),
                "num_points": len(dsarr.coords[lc]),
            },
            "V_RP": {
                "start": dsarr.coords[rc].min().item(),
                "stop": dsarr.coords[rc].max().item(),
                "num_points": len(dsarr.coords[rc]),
            },
        }

    with meas.run() as datasaver:
        scan_config = WideShotConfig(
            v_lp_config=scan_config["domain"]["V_LP"],
            v_rp_config=scan_config["domain"]["V_RP"],
        )

        def perform_measure(pixel: Tuple[int, int]) -> float:
            """Measures the current at the given pixel without saving it to the database"""
            lp, rp = scan_config.lp_range[pixel[1]], scan_config.rp_range[pixel[0]]
            station["V_LP"](lp)
            station["V_RP"](rp)
            if debug:
                point = {rc: rp, lc: lp}
                measured_value = -1 * dsarr.sel(point, method="nearest")["I_SD"].item()
            else:
                measured_value = (
                    -1 * station["I_SD"]()
                )  # contour tracing looks for peaks
            return measured_value

        wideshot_sampler = WideshotSampler(
            (scan_config.n_px_lp, scan_config.n_px_rp),
            perform_measurement=perform_measure,
        )
        wideshot_sampler.sample(on_contour_measure=lambda idx: None)

        # Save points to qcodes
        measured_data = wideshot_sampler.data
        if fill_with_threshold:
            # measured_data[~wideshot_sampler.measurement_mask] = wideshot_sampler.threshold
            measured_data[np.abs(wideshot_sampler.binary_data - 1.0) > 1e-6] = wideshot_sampler.threshold
        else:
            # binary data should be a binary mask of zero where a measurement was performed and was below threshold
            # 1 where it was performed and above threshold, and infinity everywhere else. Because there are infinities
            # in that array, we do a float point comparison and set all values that were below threshold to a background
            # level
            background_current_level, _ = mode(wideshot_sampler.calib_scan_measurement.flatten())
            measured_data[np.abs(wideshot_sampler.binary_data - 1.0) > 1e-6] = background_current_level[0]
        for point in np.ndindex(measured_data.shape):
            point = {
                "V_LP": scan_config.lp_range[point[1]],
                "V_RP": scan_config.rp_range[point[0]],
                "I_SD": -1.0*measured_data[point], # the sampler stores the negative current, so invert it back
            }
            save_measurement(point, datasaver, station)
    if debug:
        create_debug_plots_for_wideshot_sampler(
            wideshot_sampler, -dsarr["I_SD"].data, f"{os.getcwd()}/test"
        )

    with open(f"{data_save_path}/wideshot_samplers/wideshot_sampler_{datasaver.dataset.guid}.pkl", "wb") as outfile:
        wideshot_sampler.perform_measurement = None  # remove the measurement function that is not pickleable
        pickle.dump(wideshot_sampler, outfile)
    return datasaver.dataset

def save_measurement(point: Dict[str, float], datasaver: Any, station: Any) -> None:
    """Saves a dictionary to the datasaver."""
    results = []
    for point_name, point_value in point.items():
        results.append((station[point_name], point_value))
    datasaver.add_result(*results)


def _get_1d_scan_args(
    param: ParameterBase, scan_config: LinSweepConfig
) -> AbstractSweep:
    return LinSweep(param, **scan_config)


def _scan_nd(
    station: qc.Station,
    scan_config: ScanConfig,
    prepared_measurement: bool = True,
) -> DataSetProtocol:
    """Run a nd scan with bounds given by scan_config."""
    assert prepared_measurement, "Please prepare measurement before scanning"
    (
        dataset,
        _,
        _,
    ) = dond(
        *(
            _get_1d_scan_args(station[param_name], param_scan_config)
            for param_name, param_scan_config in scan_config.get("domain", {}).items()
        ),
        *(station[meas_name] for meas_name in scan_config.get("measure", [])),
        **scan_config.get("scan_kwargs", {}),
        do_plot=False,
    )
    return dataset


def parse_value_setting(param: ParameterBase, value: float) -> float:
    """Checks the value is within the bounds of param. If not, it clips it to the bounds."""
    validator = param.vals
    try:
        param.validate(value)
        return value
    except ValueError as ve:
        logger.warning(str(ve))
    if not isinstance(validator, Numbers):
        raise ValueError(f"Cannot fix value error of {param.name}.")
    clipped_value = np.clip(
        value, validator.min_value + EPSILON, validator.max_value - EPSILON
    )
    logger.warning(f"Clipping the value {param.name}: {value} to {clipped_value}")
    return clipped_value


@retry(stop=stop_after_attempt(3))
def safe_value_setting(param: ParameterBase, value: float) -> None:
    """First stops execution and asks the user if they wish to continue.
    If yes, sets the parameter to the value.
    """
    answer = input(f"Do you want to set {param.name} to {value}? [Y/n]")
    if answer.lower() == "y" or answer == "":
        param(value)
        return
    if answer.lower() == "n":
        sys.exit("Aborting.")
    logger.error("Invalid answer. Please enter [y/n].")
    raise ValueError


def validate_params_in_station(station: qc.Station, params: Sequence[str]) -> None:
    for param_name in params:
        if param_name not in station.components:
            raise ValueError(
                f"The parameter with name {param_name} has not been added to the station."
                " Please add it with `station.add_component`"
            )


class MeasurementTaker(abc.ABC):
    """Abstract class for taking scan measurements. To use it, inherit from it and implement _prepare_experiment method.
    _prepare_experiment will have access to self._experiment_config and is meant to be used to set up the experiment, e.g. set up the magnet.
    """

    def __init__(
        self,
        station: qc.Station,
        experiment_config: Dict[str, Any] = None,
        safe_mode: bool = False,
        snake_mode: bool = False,
        debug: bool = False,
    ) -> None:
        self.station = station
        self._experiment_prepared = False
        self._experiment_config = experiment_config or {}
        self.safe_mode = safe_mode
        self.snake_mode = snake_mode
        self.debug = debug

    @abc.abstractmethod
    def _prepare_experiment(self) -> None:
        """Prepare experiment, e.g. set up the magnet."""
        pass

    def __call__(self, scan_config: ScanConfig) -> DataSetProtocol:
        """Take a scan measurement and return corresponding qcodes dataset."""
        # TODO: technically jump is part of prepare experiment but mag ramp doesnt need to happen on each jump. Review abstractions
        config_keys = ["domain", "measure", "context_values"]
        param_names_used = {
            param_name
            for config_key in config_keys
            for param_name in scan_config.get(config_key, {})
        }
        validate_params_in_station(self.station, param_names_used)
        if not self._experiment_prepared:
            self._prepare_experiment()
            self._experiment_prepared = True
        self.jump(scan_config.get("context_values", {}))
        if self.snake_mode:
            return _scan_nd_with_snake(
                self.station, scan_config=scan_config, debug=self.debug
            )
        return _scan_nd(self.station, scan_config=scan_config)

    def jump(self, context_values: Dict[str, float]) -> None:
        """Set context parameters to given values."""
        # TODO: This jump function could be made so that jumps are smoother for large changes
        validate_params_in_station(self.station, context_values)
        for param_name, param_value in context_values.items():
            value = parse_value_setting(self.station[param_name], param_value)
            if self.safe_mode:
                safe_value_setting(self.station[param_name], value)
                continue
            self.station[param_name](value)

    def reset(self):
        self._experiment_prepared = False


class WideScanTaker(MeasurementTaker):
    """Take a wide scan measurement."""

    def _prepare_experiment(self) -> None:
        """Set magnetic field to the value specified in the experiment config and stop AWG."""
        ramp_magnet_before_msmt(
            self.station.IPS, self._experiment_config.get("magnetic_field", 0)
        )
        self.station.awg.stop()


class EDSRCheck(MeasurementTaker):
    def _prepare_experiment(self):
        bias_direction = self._experiment_config.get("bias_direction", "positive_bias")
        # cannot optimize directly on seconds (too small) so we use ns
        burst_time = self._experiment_config.get("burst_time_ns", 1) * 1e-9
        self.station.VS_freq(self._experiment_config["freq_vs"])
        self.station.V_LP(self._experiment_config["V_LP"])
        self.station.V_RP(self._experiment_config["V_RP"])
        # I guess this is redundant if we sweep the magnetic field
        start_pulsing(self.station.awg, get_compensated_gates(self.station), bias_direction, burst_time)
        if self._experiment_config.get("init", False):
            ramp_magnet_before_msmt(
                self.station.IPS, self._experiment_config.get("magnetic_field", 0)
            )
            self.station.LITC(self._experiment_config.get("lockin_tc"))
            self.station.mfli.sigins[0].autorange(1)
        time.sleep(self._experiment_config.get("lockin_tc") * 3)


if __name__ == "__main__":
    import os

    from bias_triangle_detection.qcodes_db_utils import query_datasets

    task_id_high_res = "cddeb475-85bd-4b68-8359-732e580522c5"

    path = "data/GeSiNW_Qubit_VTI01_Jonas_2.db"
    if not os.path.exists(path):
        print(f"! Data not found: {path}")
        sys.exit(-1)
    datasets = query_datasets(path, lambda ds: ds.name.endswith(task_id_high_res))
    from experiment_control.dummy_init import initialise_dummy_experiment

    station = initialise_dummy_experiment()

    # WIDE SCAN EXAMPLE
    experiment_setup_wide = {"magnetic_field": 0}
    coulomb_peak_scanner = WideScanTaker(
        station, experiment_setup_wide, snake_mode=True
    )

    # test mode no exp preparation
    coulomb_peak_scanner._prepare_experiment = lambda: None

    dsarr = datasets[0].to_xarray_dataset()
    rc, lc = list(dsarr.dims)

    # n_points_plunges = 66
    scan_config_wide = {
        "domain": {
            "V_LP": {
                "start": dsarr.coords[lc].min().item(),
                "stop": dsarr.coords[lc].max().item(),
                "num_points": len(dsarr.coords[lc]),
            },
            "V_RP": {
                "start": dsarr.coords[rc].min().item(),
                "stop": dsarr.coords[rc].max().item(),
                "num_points": len(dsarr.coords[rc]),
            },
        },
        "measure": ["I_SD"],
        "context_values": {"V_L": 1.350, "V_M": 0.660, "V_R": 1.020},
        "scan_kwargs": {"measurement_name": "coarse_tuning_snake"},
    }
    dataset = coulomb_peak_scanner(scan_config_wide)
    assert isinstance(dataset, DataSetProtocol)
    # name_data = station.I_SD.name
    # assert set(dataset.parameters.split(",")) == {
    #     station.V_LP.full_name,
    #     station.V_RP.full_name,
    #     name_data,
    # }
    # assert dataset.get_parameter_data(name_data)[name_data][name_data].shape == (
    #     n_points_plunges,
    #     n_points_plunges,
    # )

    # # EDSR CHECK EXAMPLE
    # experiment_setup_edsr = {
    #     "bias_direction": "positive_bias",  # candidate
    #     "burst_time": 2e-9,
    #     "freq_vs": 2.79e9,
    #     "left_plunger_voltage": 0,  # candidate
    #     "right_plunger_voltage": 0,  # candidate
    #     "magnetic_field": 0,
    #     "lockin_tc": 2,
    # }
    # edsr_check_taker = EDSRCheck(station, experiment_setup_edsr)
    # edsr_check_taker._prepare_experiment = lambda: None
    # dataset_name = "edsr_check_during_optim"
    # scan_config_edsr = {
    #     "domain": {
    #         "field_setpoint": {"start": 0, "stop": 0.5, "num_points": 100}
    #     },  # 0.1 window around candidate
    #     "measure": ["LIX", "LIY"],
    #     "context_values": {"V_L": 1.350, "V_M": 0.660, "V_R": 1.020},
    #     "scan_kwargs": {"measurement_name": dataset_name},
    # }
    # dataset = edsr_check_taker(scan_config_edsr)
    # assert dataset.name == dataset_name
