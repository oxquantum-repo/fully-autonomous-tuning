from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import qcodes as qc
from bias_triangle_detection.bayesian_optimization.bayesian_optimizer import (
    EvaluationResult,
)
from bias_triangle_detection.coord_change import EuclideanTransformation
from qcodes.dataset import DataSetProtocol
from qcodes.dataset.data_set_protocol import DataSetProtocol

from experiment_control.data_taking_utils import EDSRCheck, ScanConfig


class EDSRPeakScanner:
    def __init__(
        self,
        station: qc.Station,
        scan_config: ScanConfig,
        experiment_config: Dict[str, Any] = None,
        safe_mode: bool = False,
    ) -> None:
        self.scan_taker = EDSRCheck(
            station,
            experiment_config=experiment_config,
            safe_mode=safe_mode,
        )
        self.transform = None
        self.scan_config = scan_config
        self.target_coords = None
        self.first_measurement = True

    def set_transform(
        self,
        centroid: np.ndarray,
        angle: float,
        orig_coords: List[str],
        target_coords: List[str],
    ) -> None:
        """From surface points we get euclidean transformation to optim region."""
        self.target_coords = target_coords
        angle = np.deg2rad(-angle)
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        self.transform = EuclideanTransformation(
            rotation_matrix, centroid, orig_coords, target_coords
        )

    def optim_point_to_voltage_point(
        self, optim_point: Dict[str, float]
    ) -> Dict[str, float]:
        rectangle_point = {name: optim_point.pop(name) for name in self.target_coords}
        voltage_point = self.transform.inverse(rectangle_point)
        optim_point.update(voltage_point)
        return optim_point

    def __call__(self, optim_point_rect: Dict[str, float]) -> DataSetProtocol:
        optim_point = self.optim_point_to_voltage_point(optim_point_rect)
        self.scan_taker._experiment_config.update(optim_point)
        self.scan_taker.reset()
        dataset = self.scan_taker(self.scan_config)
        if self.first_measurement:
            self.scan_taker._experiment_config["init"] = False
            self.first_measurement = False
        self.scan_config = flip_scan_order(self.scan_config)
        return dataset

    def _readout_data(self, dataset: DataSetProtocol) -> np.ndarray:
        # rows are magnetic field, columns are lockins
        return dataset.to_xarray_dataset().to_array().to_numpy().T


def flip_scan_order(scan_config: ScanConfig) -> ScanConfig:
    assert len(scan_config["domain"]) == 1
    scan_var_name = next(iter(scan_config["domain"]))
    scan_config_start = scan_config["domain"][scan_var_name].pop("start")
    scan_config_stop = scan_config["domain"][scan_var_name].pop("stop")
    scan_config["domain"][scan_var_name]["start"] = scan_config_stop
    scan_config["domain"][scan_var_name]["stop"] = scan_config_start
    return scan_config


@dataclass
class Score:
    scorer: Callable[[np.ndarray], float]
    edsr_scanner: EDSRPeakScanner
    metadata: Dict[str, Any]

    def _on_scoring_dataset(
        self, dataset: DataSetProtocol, point: Dict[str, float]
    ) -> None:
        optim_readout_point = self.edsr_scanner.optim_point_to_voltage_point(
            point.copy()
        )
        optim_readout_point.update(point)
        for k, v in {**optim_readout_point, **self.metadata}.items():
            dataset.add_metadata(k, v)

    def __call__(self, **point: float) -> EvaluationResult:
        readout_dataset = self.edsr_scanner(point.copy())
        readout_data = self.edsr_scanner._readout_data(readout_dataset)
        score = self.scorer(readout_data)
        self._on_scoring_dataset(readout_dataset, point)
        return EvaluationResult(
            score=-1 * score, config=point, metadata={"guid": readout_dataset.guid}
        )
