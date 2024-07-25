import os


# qcodes imports
import qcodes as qc
from qcodes import (
    initialise_or_create_database_at,
    load_or_create_experiment,
)

from qcodes.instrument_drivers.mock_instruments import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)


def initialise_experiment(
        db_name="test_database.db",  # Database name
        sample_name="test_sample",  # Sample name
        exp_name="test_exp_name",  # Experiment name
    ):
    db_file_path = os.path.join(os.getcwd(), db_name)
    qc.config.core.db_location = db_file_path
    initialise_or_create_database_at(db_file_path)

    experiment = load_or_create_experiment(experiment_name=exp_name,
                                           sample_name=sample_name)

    station = qc.Station()

    # A dummy signal generator with two parameters ch1 and ch2
    dac = DummyInstrument('dac', gates=['ch1', 'ch2'])
    station.add_component(dac)

    # A dummy digital multimeter that generates a synthetic data depending
    # on the values set on the setter_instr, in this case the dummy dac
    dmm = DummyInstrumentWithMeasurement('dmm', setter_instr=dac)
    station.add_component(dmm)


    return station


if __name__=='__main__':
    station = initialise_experiment()
