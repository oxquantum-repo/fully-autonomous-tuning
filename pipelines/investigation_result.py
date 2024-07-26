from dataclasses import dataclass, field
from typing import Dict


@dataclass
class InvestigationResult:
    guids: Dict[str, str] = field(default_factory=dict)
    rabi_frequency: float = None
    g_factor: float = None
    magnetic_field: float = None
    burst_time_guess: float = None
    barrier_voltages: Dict[str, float] = field(default_factory=dict)
    centroid_triangle: Dict[str, float] = field(default_factory=dict)
    readout_spot: Dict[str, float] = field(default_factory=dict)
    readout_lost: bool = False


@dataclass
class InvestigationResultCurrentOptimisation:
    guids: Dict[str, str] = field(default_factory=dict)
    current_ratio: float = None
    barrier_voltages: Dict[str, float] = field(default_factory=dict)
    centroid_triangle: Dict[str, float] = field(default_factory=dict)
    readout_lost: bool = False

@dataclass
class InvestigationResultRabiOptimisation:
    guids: Dict[str, str] = field(default_factory=dict)
    rabi_frequency: float = None
    quality_factor: float = None
    g_factor: float = None
    magnetic_field: float = None
    burst_time_guess: float = None
    barrier_voltages: Dict[str, float] = field(default_factory=dict)
    plunger_voltages: Dict[str, float] = field(default_factory=dict)
    plunger_voltages_transformed: Dict[str, float] = field(default_factory=dict)
    centroid_triangle: Dict[str, float] = field(default_factory=dict)
    readout_lost: bool = False