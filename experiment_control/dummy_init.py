# Adding this temporarily until MR with qcodes_addons as package
import sys

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


def initialise_dummy_experiment(db_file_path: str = "data/dummy_db.db"):
    """Initialise a dummy experiment with a dummy station and dummy instruments. Meant for integration tests."""
    initialise_or_create_database_at(db_file_path)

    sample_name = "Butch"  # Sample name
    exp_name = "Qubit_Search"  # Experiment name

    load_or_create_experiment(experiment_name=exp_name, sample_name=sample_name)

    station = qc.Station()
    dac = DummyInstrument(
        name="LNHR_dac", gates=["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]
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
    field_setpoint = GateParameter(
        dac.ch6, name="field_setpoint", unit="T", value_range=value_range
    )
    components = [ISD, VLP, VRP, VL, VM, VR, LIX, LIY, field_setpoint, LIR, LIPhi]
    for component in components:
        station.add_component(component)
    return station
